from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import MiniBatchDictionaryLearning, sparse_encode
from sklearn.utils.validation import check_is_fitted, validate_data
from sklearn.utils import check_random_state
import numpy as np
import scipy.sparse as sp
import torch
from tqdm.auto import tqdm

from ._dictionary_learning_torch import fista_lasso, fit_dictionary_torch, lipschitz_constant

_BACKENDS = ("auto", "sklearn", "torch")


def _mean_squared_norm(X: np.ndarray, chunk: int = 65536) -> float:
    """mean_i ||x_i||^2, accumulated in float64 over row chunks (no copy of X)."""
    total = 0.0
    for start in range(0, len(X), chunk):
        block = X[start:start + chunk]
        total += float(np.einsum("ij,ij->", block, block, dtype=np.float64))
    return total / len(X)


class MiniBatchDictionaryCoding(TransformerMixin, BaseEstimator):
    """
    L1 sparse coding with a learned dictionary. Mirrors SparseAutoencoder /
    LocalityConstrainedCoding's fit/transform API.

    Solves min 0.5 ||s x - mean - c D||^2 + alpha ||c||_1 per patch, where s is
    `scale_factor_` (1 unless scale=True). With
    positive_code=True, codes are nonnegative like SAE ReLU activations; with
    positive_dict=False, atoms are signed. Inputs are centered with the
    training mean (`center=True`) because, unlike an SAE, the model has no bias.

    Two backends solve the same objective:
      - "sklearn": sklearn.decomposition.MiniBatchDictionaryLearning for fit and
        sparse_encode(transform_algorithm) for transform. CPU only.
      - "torch": a port of sklearn's online dictionary update with batched FISTA
        as the sparse coder (see _dictionary_learning_torch). Runs on any torch
        device.
    `components_` is shared, so a model fit with one backend can transform
    with the other.

    Parameters
    ----------
    n_components, alpha, fit_algorithm, transform_algorithm, transform_alpha,
    positive_code, positive_dict, batch_size, max_iter, n_jobs
        As in MiniBatchDictionaryLearning. transform_alpha=None defaults to
        alpha. Tune alpha to reach a target mean number of active features per
        patch (see `_training_log["mean_l0"]` after fit). fit_algorithm and
        n_jobs apply to the sklearn backend only; fit_algorithm defaults to "cd"
        because sklearn rejects positive_code=True with fit_algorithm="lars".
        The torch backend requires transform_algorithm "lasso_lars" or
        "lasso_cd" (the lasso objective).
    max_steps : int, optional
        Total number of minibatch steps (torch backend only). Overrides
        max_iter, e.g. to match another model's training steps.
    center : bool, default=True
        Subtract the training mean (of the scaled data) in fit and transform.
    scale : bool, default=False
        Multiply inputs by one constant, `scale_factor_ = sqrt(1 / mean ||x||^2)`,
        so the scaled training data has unit mean squared norm. Makes alpha and
        code magnitudes comparable across embedding models. SparseAutoencoder's
        scaling gives mean ||x||^2 = sqrt(d) instead, so its scores are larger
        by exactly d**0.25 at the same fit.
    transform_batch_size : int, default=65536
        Rows encoded per block in transform; each block is sparsified before
        the next, bounding memory for the dense code.
    device : str, optional
        Default torch device when fit/transform get none; if both are None,
        "cuda" if available, else "cpu".
    backend : {"auto", "sklearn", "torch"}, default="auto"
        "auto" uses sklearn on "cpu" and torch on any other device.
    fista_max_iter : int, default=1000
    fista_tol : float, default=1e-4
        FISTA stops when the KKT residual is <= fista_tol * alpha.
    dict_kwargs : dict, optional
        Extra keyword arguments for MiniBatchDictionaryLearning (e.g. `tol`,
        `max_no_improvement`, `transform_max_iter`). The torch backend reads
        `tol` and `max_no_improvement`.
    random_state : int, RandomState or None
    """

    def __init__(self,
                 n_components: int = 1024,
                 alpha: float = 1.0,
                 fit_algorithm: str = "cd",
                 transform_algorithm: str = "lasso_lars",
                 transform_alpha: "float | None" = None,
                 positive_code: bool = True,
                 positive_dict: bool = False,
                 batch_size: int = 2048,
                 max_iter: int = 50,
                 max_steps: "int | None" = None,
                 center: bool = True,
                 scale: bool = False,
                 transform_batch_size: int = 65536,
                 n_jobs: "int | None" = -1,
                 device: "str | None" = None,
                 backend: str = "auto",
                 fista_max_iter: int = 1000,
                 fista_tol: float = 1e-4,
                 dict_kwargs: "dict | None" = None,
                 random_state=None):
        self.n_components = n_components
        self.alpha = alpha
        self.fit_algorithm = fit_algorithm
        self.transform_algorithm = transform_algorithm
        self.transform_alpha = transform_alpha
        self.positive_code = positive_code
        self.positive_dict = positive_dict
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.max_steps = max_steps
        self.center = center
        self.scale = scale
        self.transform_batch_size = transform_batch_size
        self.n_jobs = n_jobs
        self.device = device
        self.backend = backend
        self.fista_max_iter = fista_max_iter
        self.fista_tol = fista_tol
        self.dict_kwargs = dict_kwargs
        self.random_state = random_state

    def _resolve_backend(self, device) -> "tuple[str, str]":
        """(device, backend) for a fit/transform call."""
        if self.backend not in _BACKENDS:
            raise ValueError(f"backend must be one of {_BACKENDS}, got {self.backend!r}")
        device = device or self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        backend = self.backend
        if backend == "auto":
            backend = "sklearn" if torch.device(device).type == "cpu" else "torch"
        if backend == "torch" and self.transform_algorithm not in ("lasso_lars", "lasso_cd"):
            raise ValueError(
                f"The torch backend solves the lasso objective only; transform_algorithm="
                f"{self.transform_algorithm!r} is not supported. Use 'lasso_lars' or 'lasso_cd'."
            )
        return str(device), backend

    def fit(self, X, y=None, *,
            obsm_key=None, tile_key="tiles",
            device=None,
            fraction: float = 1.0,
            sample_size: "int | None" = None,
            verbose: "int | bool" = False):
        if obsm_key is not None:
            from mesoslide._slides import SlideSource
            source = SlideSource(X, tile_key=tile_key)
            X = np.vstack([table.obsm[obsm_key] for _, table in source])

        device, backend = self._resolve_backend(device)
        self.random_state_ = check_random_state(self.random_state)
        X = validate_data(self, X, accept_sparse=False, dtype=[np.float64, np.float32])
        assert len(X.shape) == 2  # expect X shape = B x d_emb
        if not 0 < fraction <= 1:
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")
        if fraction < 1:
            n_samples = round(fraction * len(X))
            idx = self.random_state_.choice(len(X), size=n_samples, replace=False)
            X = X[idx]
        if sample_size is not None and sample_size < len(X):
            idx = self.random_state_.choice(len(X), size=sample_size, replace=False)
            X = X[idx]

        self.scale_factor_ = (float(np.sqrt(1.0 / _mean_squared_norm(X)))
                              if self.scale else 1.0)
        self.mean_ = (X.mean(axis=0) * self.scale_factor_ if self.center
                      else np.zeros(X.shape[1], dtype=X.dtype)).astype(X.dtype)
        self.embed_dim_ = self.n_components
        seed = int(self.random_state_.randint(0, 2**31 - 1))
        dict_kwargs = self.dict_kwargs or {}

        if backend == "sklearn":
            if self.max_steps is not None:
                raise ValueError("max_steps is only supported by the torch backend; "
                                 "use max_iter (epochs) with the sklearn backend.")
            dl = MiniBatchDictionaryLearning(
                n_components=self.n_components,
                alpha=self.alpha,
                fit_algorithm=self.fit_algorithm,
                transform_algorithm=self.transform_algorithm,
                transform_alpha=self.transform_alpha,
                positive_code=self.positive_code,
                positive_dict=self.positive_dict,
                batch_size=self.batch_size,
                max_iter=self.max_iter,
                n_jobs=self.n_jobs,
                random_state=seed,
                verbose=verbose,
                **dict_kwargs,
            ).fit(X * X.dtype.type(self.scale_factor_) - self.mean_)
            self.components_ = dl.components_
            log = {"n_iter": float(dl.n_iter_), "n_steps": dl.n_steps_}
        else:
            self.components_, log = fit_dictionary_torch(
                X,
                n_components=self.n_components,
                alpha=self.alpha,
                batch_size=self.batch_size,
                max_iter=self.max_iter,
                max_steps=self.max_steps,
                tol=dict_kwargs.get("tol", 1e-3),
                max_no_improvement=dict_kwargs.get("max_no_improvement", 10),
                positive_code=self.positive_code,
                positive_dict=self.positive_dict,
                fista_max_iter=self.fista_max_iter,
                fista_tol=self.fista_tol,
                device=device,
                seed=seed,
                mean=self.mean_,
                scale=self.scale_factor_,
                verbose=verbose,
            )

        # Mean active features per patch on a training subsample, for tuning alpha.
        n_eval = min(len(X), 10000)
        eval_idx = self.random_state_.choice(len(X), size=n_eval, replace=False)
        codes = self.transform(X[eval_idx], device=device)
        mean_l0 = float(np.diff(codes.indptr).mean())
        self._training_log = {**log, "mean_l0": mean_l0, "backend": backend, "device": device}
        if verbose:
            print(f"Mean active features per patch: {mean_l0:.2f}")
        return self

    def transform(self, X, column_keep_indices=None, device=None, *,
                  obsm_key=None, tile_key="tiles", sparse_key_added=None,
                  overwrite=False, save=True, progress_bar=False):
        if obsm_key is not None:
            from mesoslide.tools._feature_extraction import _write_sparse_features

            slides = X if isinstance(X, (list, tuple)) else [X]
            table_key = f"{tile_key}_table"
            sparse_key = sparse_key_added or f"{obsm_key}_dl"
            for slide in slides:
                table = slide.tables[table_key]
                if overwrite or not any(
                    v.startswith(f"{sparse_key}_") for v in table.var_names
                ):
                    matrix = self.transform(
                        table.obsm[obsm_key], column_keep_indices, device,
                        progress_bar=progress_bar,
                    )
                    table = _write_sparse_features(table, sparse_key, matrix)
                    slide.tables[table_key] = table
                    if save:
                        slide.write_element(table_key)
            return X

        check_is_fitted(self)
        X = validate_data(self, X, accept_sparse=False, reset=False,
                          dtype=[np.float64, np.float32])
        device, backend = self._resolve_backend(device)
        n_features = self.embed_dim_ if column_keep_indices is None else len(column_keep_indices)
        if len(X) == 0:
            return sp.csr_matrix((0, n_features), dtype=np.float32)

        alpha = self.alpha if self.transform_alpha is None else self.transform_alpha
        starts = range(0, len(X), self.transform_batch_size)
        blocks = []
        if backend == "sklearn":
            max_iter = (self.dict_kwargs or {}).get("transform_max_iter", 1000)
            for start in tqdm(starts, disable=not progress_bar):
                block = X[start:start + self.transform_batch_size] * self.scale_factor_ - self.mean_
                # Lasso codes depend on every atom, so encode fully, then slice.
                code = sparse_encode(block, self.components_.astype(block.dtype, copy=False),
                                     algorithm=self.transform_algorithm, alpha=alpha,
                                     max_iter=max_iter, n_jobs=self.n_jobs,
                                     positive=self.positive_code)
                if column_keep_indices is not None:
                    code = code[:, column_keep_indices]
                blocks.append(sp.csr_matrix(code.astype(np.float32)))
            return sp.vstack(blocks, format="csr")

        dtype = torch.float64 if X.dtype == np.float64 else torch.float32
        D = torch.as_tensor(self.components_, dtype=dtype, device=device)
        mean = torch.as_tensor(self.mean_, dtype=dtype, device=device)
        G = D @ D.T
        L = lipschitz_constant(G)
        with torch.no_grad():
            for start in tqdm(starts, disable=not progress_bar):
                block = torch.as_tensor(X[start:start + self.transform_batch_size],
                                        dtype=dtype).to(device) * self.scale_factor_ - mean
                code = fista_lasso(block, D, alpha, positive=self.positive_code,
                                   max_iter=self.fista_max_iter, tol=self.fista_tol, G=G, L=L)
                if column_keep_indices is not None:
                    code = code[:, column_keep_indices]
                arr = code.to(torch.float32).to_sparse().cpu()
                row_ind, col_ind = arr.indices().numpy()
                blocks.append(sp.csr_matrix((arr.values().numpy(), (row_ind, col_ind)),
                                            shape=(block.shape[0], n_features)))
        return sp.vstack(blocks, format="csr")

    def inverse_transform(self, Xt) -> np.ndarray:
        """Reconstruct descriptors from codes (x_hat = (codes @ D + mean) / scale_factor_)."""
        check_is_fitted(self)
        return (np.asarray(Xt @ self.components_) + self.mean_) / self.scale_factor_
