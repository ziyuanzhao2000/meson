"""Fuzzy c-means soft clustering with the sparse-coding fit/transform interface.

Ported from soft-clustering's `FuzzyCMeans` (soft_clustering/_fcm.py, MIT licence), with the same
algorithm and numerics (log-space membership rule, eps guards, k-means++ seeding, objective
J(U, centers) evaluated after each center update). Changes from the reference:

- convergence uses a relative tolerance, |J_prev - J| <= tol * J, since an absolute tolerance never
  triggers on large data;
- `transform` assigns memberships to new samples from the fitted centers;
- no global random seeding; k-means++ keeps a running minimum distance instead of recomputing
  distances to all chosen centers (same draws up to floating-point rounding);
- distances are computed in row blocks, and a torch backend runs the same updates on any device.
"""
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data
from tqdm.auto import tqdm

from ._dictionary_learning import _mean_squared_norm

_BACKENDS = ("auto", "numpy", "torch")
_EPS = 1e-12


# ---------------------------------------------------------------------------
# numpy kernels (float64, as the reference)
# ---------------------------------------------------------------------------

def _dist2_np(X, centers, batch_size):
    """Squared Euclidean distances (n, K), computed in row blocks in float64."""
    centers = np.asarray(centers, dtype=np.float64)
    c_norm = np.sum(centers * centers, axis=1)
    out = np.empty((len(X), len(centers)), dtype=np.float64)
    for start in range(0, len(X), batch_size):
        block = np.asarray(X[start:start + batch_size], dtype=np.float64)
        d = np.sum(block * block, axis=1, keepdims=True) + c_norm[None, :] - 2.0 * (block @ centers.T)
        np.maximum(d, 0.0, out=d)
        out[start:start + len(block)] = d
    return out


def _memberships_np(dist2, m):
    """u_ik = d_ik^-p / sum_j d_ij^-p with p = 1/(m-1), in log space (reference ratio_memberships)."""
    log_d = np.log(np.maximum(dist2 + _EPS, np.finfo(np.float64).tiny))
    scores = -(1.0 / (m - 1.0)) * (log_d - log_d.min(axis=1, keepdims=True))
    np.exp(scores, out=scores)
    total = scores.sum(axis=1, keepdims=True)
    U = scores / np.where(total > 0, total, 1.0)
    U = np.maximum(U, _EPS)
    return U / (U.sum(axis=1, keepdims=True) + _EPS)


def _centers_np(X, U, m, batch_size):
    Um = U ** m
    num = np.zeros((U.shape[1], X.shape[1]), dtype=np.float64)
    for start in range(0, len(X), batch_size):
        num += Um[start:start + batch_size].T @ np.asarray(X[start:start + batch_size], dtype=np.float64)
    return num / (Um.sum(axis=0)[:, None] + _EPS)


def _kmeans_pp(X, K, rng, batch_size):
    """k-means++ seeding (reference _init_centers_kpp), with a running minimum distance."""
    n = len(X)
    centers = np.empty((K, X.shape[1]), dtype=np.float64)
    centers[0] = X[rng.integers(0, n)]
    d_min = _dist2_np(X, centers[:1], batch_size)[:, 0]
    for k in range(1, K):
        s = d_min.sum()
        idx = rng.integers(0, n) if not np.isfinite(s) or s <= 0.0 else rng.choice(n, p=d_min / s)
        centers[k] = X[idx]
        np.minimum(d_min, _dist2_np(X, centers[k:k + 1], batch_size)[:, 0], out=d_min)
    return centers


# ---------------------------------------------------------------------------
# torch kernels (same math on any device)
# ---------------------------------------------------------------------------

def _dist2_torch(X, centers):
    d = (X * X).sum(1, keepdim=True) + (centers * centers).sum(1)[None, :] - 2.0 * (X @ centers.T)
    return d.clamp_min_(0.0)


def _memberships_torch(dist2, m):
    tiny = torch.finfo(dist2.dtype).tiny
    log_d = torch.log(torch.clamp_min(dist2 + _EPS, tiny))
    scores = torch.exp(-(1.0 / (m - 1.0)) * (log_d - log_d.min(dim=1, keepdim=True).values))
    total = scores.sum(1, keepdim=True)
    U = scores / torch.where(total > 0, total, torch.ones_like(total))
    U = U.clamp_min(_EPS)
    return U / (U.sum(1, keepdim=True) + _EPS)


class FuzzyCMeans(TransformerMixin, BaseEstimator):
    """
    Fuzzy c-means soft clustering. Mirrors MiniBatchDictionaryCoding's fit/transform API.

    Minimizes J = sum_i sum_k u_ik^m ||s x_i - c_k||^2 over memberships u (rows sum to 1) and
    centers c, by alternating the closed-form updates. `transform` returns each sample's
    memberships, so column k is a soft score for cluster k, usable like an SAE feature
    (`feature_extraction(sparse=True, sparse_transform=fcm.transform)`, `feature_scorer`).

    Parameters
    ----------
    n_components : int, default=30
        Number of clusters.
    m : float, default=2.0
        Fuzzifier, > 1. Larger is softer; in high dimensions, values near 2 can make all
        memberships nearly uniform, so check `_training_log` (mean max membership).
    max_iter : int, default=300
    tol : float, default=1e-6
        Stop when |J_prev - J| <= tol * J. 0 runs `max_iter` iterations.
    init : {"kmeans++", "random"} or ndarray of shape (n_components, n_features), default="kmeans++"
        Initial centers (in scaled units when an array is given).
    scale : bool, default=False
        Multiply inputs by `scale_factor_ = sqrt(1 / mean ||x||^2)`, as MiniBatchDictionaryCoding.
        Memberships are invariant to this; it only changes the units of centers and J.
    batch_size : int, default=65536
        Rows per block for distances (and for transform).
    device : str, optional
        Default torch device when fit/transform get none; if both are None, "cuda" if available.
    backend : {"auto", "numpy", "torch"}, default="auto"
        "auto" uses numpy on "cpu" and torch on any other device.
    dtype : {"float64", "float32"}, optional
        Torch precision; default float32. The numpy backend always uses float64.
    random_state : int or None
        Seeds k-means++ (shared by both backends) or the random initial memberships.
    """

    def __init__(self, n_components: int = 30, m: float = 2.0, max_iter: int = 300, tol: float = 1e-6,
                 init="kmeans++", scale: bool = False, batch_size: int = 65536,
                 device: "str | None" = None, backend: str = "auto", dtype: "str | None" = None,
                 random_state: "int | None" = None):
        self.n_components = n_components
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.scale = scale
        self.batch_size = batch_size
        self.device = device
        self.backend = backend
        self.dtype = dtype
        self.random_state = random_state

    def _resolve_backend(self, device) -> "tuple[str, str]":
        if self.backend not in _BACKENDS:
            raise ValueError(f"backend must be one of {_BACKENDS}, got {self.backend!r}")
        device = device or self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        backend = self.backend
        if backend == "auto":
            backend = "numpy" if torch.device(device).type == "cpu" else "torch"
        return str(device), backend

    def _torch_dtype(self):
        return torch.float64 if self.dtype == "float64" else torch.float32

    def _initial_centers(self, X, rng):
        """Initial centers in scaled units (float64), shared by both backends."""
        K = self.n_components
        if isinstance(self.init, np.ndarray):
            centers = np.asarray(self.init, dtype=np.float64)
            if centers.shape != (K, X.shape[1]):
                raise ValueError(f"init array must have shape {(K, X.shape[1])}, got {centers.shape}")
            return centers
        Xs = _Scaled(X, self.scale_factor_)
        if self.init == "kmeans++":
            return _kmeans_pp(Xs, K, rng, self.batch_size)
        if self.init == "random":
            U = rng.random((len(X), K))
            U = np.maximum(U, _EPS)
            U = U / (U.sum(axis=1, keepdims=True) + _EPS)
            return _centers_np(Xs, U, self.m, self.batch_size)
        raise ValueError(f"Unknown init={self.init!r}")

    def fit(self, X, y=None, *, obsm_key=None, tile_key="tiles", device=None,
            fraction: float = 1.0, sample_size: "int | None" = None, verbose: "int | bool" = False):
        if obsm_key is not None:
            from mesoslide._slides import SlideSource
            source = SlideSource(X, tile_key=tile_key)
            X = np.vstack([table.obsm[obsm_key] for _, table in source])

        device, backend = self._resolve_backend(device)
        X = validate_data(self, X, accept_sparse=False, dtype=[np.float64, np.float32])
        if self.m <= 1.0:
            raise ValueError(f"m must be > 1, got m={self.m}")
        if not 0 < fraction <= 1:
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")
        rng = np.random.default_rng(self.random_state)
        if fraction < 1:
            X = X[rng.choice(len(X), size=round(fraction * len(X)), replace=False)]
        if sample_size is not None and sample_size < len(X):
            X = X[rng.choice(len(X), size=sample_size, replace=False)]
        if not 1 <= self.n_components <= len(X):
            raise ValueError(f"n_components must be in [1, n_samples={len(X)}], got {self.n_components}")

        self.scale_factor_ = float(np.sqrt(1.0 / _mean_squared_norm(X))) if self.scale else 1.0
        self.embed_dim_ = self.n_components
        centers = self._initial_centers(X, rng)

        if backend == "numpy":
            U, centers, trajectory = self._fit_numpy(_Scaled(X, self.scale_factor_), centers, verbose)
        else:
            U, centers, trajectory = self._fit_torch(X, centers, device, verbose)

        self.cluster_centers_ = centers  # float64, scaled units
        self.n_iter_ = len(trajectory)
        self.objective_trajectory_ = np.asarray(trajectory, dtype=np.float64)
        top = U.max(axis=1)
        entropy = -(U * np.log(np.maximum(U, _EPS))).sum(axis=1) / np.log(self.n_components)
        self._training_log = {
            "n_iter": self.n_iter_, "objective": float(trajectory[-1]) if trajectory else np.nan,
            "mean_max_membership": float(top.mean()), "mean_normalized_entropy": float(entropy.mean()),
            "cluster_sizes": np.bincount(U.argmax(axis=1), minlength=self.n_components).tolist(),
            "backend": backend, "device": device,
        }
        if verbose:
            print(f"FCM: {self.n_iter_} iterations, mean max membership {top.mean():.3f}")
        return self

    def _fit_numpy(self, Xs, centers, verbose):
        trajectory, obj_prev, U = [], np.inf, None
        for it in range(self.max_iter):
            U = _memberships_np(_dist2_np(Xs, centers, self.batch_size), self.m)
            centers = _centers_np(Xs, U, self.m, self.batch_size)
            obj = float(np.sum((U ** self.m) * _dist2_np(Xs, centers, self.batch_size)))
            trajectory.append(obj)
            if verbose and it % 10 == 0:
                print(f"iter {it}: J = {obj:.6g}")
            if abs(obj_prev - obj) <= self.tol * obj:
                break
            obj_prev = obj
        return U, centers, trajectory

    def _fit_torch(self, X, centers, device, verbose):
        dtype = self._torch_dtype()
        with torch.no_grad():
            Xt = torch.as_tensor(X, device=device).to(dtype) * self.scale_factor_
            C = torch.as_tensor(centers, dtype=dtype, device=device)
            trajectory, obj_prev, U = [], np.inf, None
            for it in range(self.max_iter):
                U = torch.cat([_memberships_torch(_dist2_torch(b, C), self.m)
                               for b in Xt.split(self.batch_size)])
                Um = U ** self.m
                C = (Um.T @ Xt) / (Um.sum(0)[:, None] + _EPS)
                obj = float(sum((Um_b * _dist2_torch(b, C)).sum(dtype=torch.float64)
                                for Um_b, b in zip(Um.split(self.batch_size), Xt.split(self.batch_size))))
                trajectory.append(obj)
                if verbose and it % 10 == 0:
                    print(f"iter {it}: J = {obj:.6g}")
                if abs(obj_prev - obj) <= self.tol * obj:
                    break
                obj_prev = obj
            return U.double().cpu().numpy(), C.double().cpu().numpy(), trajectory

    def transform(self, X, column_keep_indices=None, device=None, *,
                  obsm_key=None, tile_key="tiles", sparse_key_added=None,
                  overwrite=False, save=True, progress_bar=False):
        """Memberships (n, n_components) of samples given the fitted centers, float32; rows sum to 1."""
        if obsm_key is not None:
            from mesoslide.tools._feature_extraction import _write_sparse_features

            slides = X if isinstance(X, (list, tuple)) else [X]
            table_key = f"{tile_key}_table"
            sparse_key = sparse_key_added or f"{obsm_key}_fcm"
            for slide in slides:
                table = slide.tables[table_key]
                if overwrite or not any(v.startswith(f"{sparse_key}_") for v in table.var_names):
                    matrix = self.transform(table.obsm[obsm_key], column_keep_indices, device,
                                            progress_bar=progress_bar)
                    table = _write_sparse_features(table, sparse_key, sp.csr_matrix(matrix))
                    slide.tables[table_key] = table
                    if save:
                        slide.write_element(table_key)
            return X

        check_is_fitted(self)
        X = validate_data(self, X, accept_sparse=False, reset=False, dtype=[np.float64, np.float32])
        device, backend = self._resolve_backend(device)
        cols = slice(None) if column_keep_indices is None else column_keep_indices
        starts = range(0, len(X), self.batch_size)
        if backend == "numpy":
            Xs = _Scaled(X, self.scale_factor_)
            out = [_memberships_np(_dist2_np(Xs[s:s + self.batch_size], self.cluster_centers_, self.batch_size),
                                   self.m)[:, cols] for s in tqdm(starts, disable=not progress_bar)]
        else:
            dtype = self._torch_dtype()
            C = torch.as_tensor(self.cluster_centers_, dtype=dtype, device=device)
            with torch.no_grad():
                out = [_memberships_torch(_dist2_torch(
                           torch.as_tensor(X[s:s + self.batch_size]).to(device, dtype) * self.scale_factor_, C),
                           self.m)[:, cols].cpu().numpy() for s in tqdm(starts, disable=not progress_bar)]
        if not out:
            n_cols = self.n_components if column_keep_indices is None else len(column_keep_indices)
            return np.zeros((0, n_cols), dtype=np.float32)
        return np.concatenate(out).astype(np.float32)

    def predict(self, X, device=None) -> np.ndarray:
        """Hard cluster assignment: index of the largest membership."""
        return self.transform(X, device=device).argmax(axis=1)


class _Scaled:
    """Row-sliceable view of X multiplied by a constant, without copying X."""

    def __init__(self, X, factor):
        self.X, self.factor, self.shape = X, factor, X.shape

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        block = np.asarray(self.X[idx], dtype=np.float64)
        return block * self.factor if self.factor != 1.0 else block
