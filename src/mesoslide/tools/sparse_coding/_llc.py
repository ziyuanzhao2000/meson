import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data
from sklearn.utils import check_random_state
import numpy as np
import scipy.sparse as sp
from tqdm.auto import tqdm

from ._kmeans_backends import fit_kmeans_backend


class LLCModel(nn.Module):
    """
    Holds an LLC codebook and exposes locality-constrained linear coding as
    encode/decode operations.

    Unlike SimpleAutoencoder, there is no learned encoder: `encode` is a
    non-parametric K-nearest-neighbor search plus a constrained least-squares
    solve against the current codebook, following Wang et al., "Locality-
    constrained Linear Coding for Image Classification" (CVPR 2010).
    """

    def __init__(self, input_dim: int, n_codewords: int = 2048):
        super().__init__()
        self.input_dim = input_dim
        self.n_codewords = n_codewords
        # Parameter (not a buffer) with requires_grad=False: leaves the door
        # open for a future learned-codebook refinement (the paper's
        # Algorithm 4.1, out of scope here) without restructuring this class.
        self.codebook = nn.Parameter(
            torch.empty(n_codewords, input_dim), requires_grad=False
        )

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """codes: (N, n_codewords) -> (N, input_dim)."""
        return codes @ self.codebook

    def encode(self, x: torch.Tensor, n_neighbors: int = 5,
               constraint: str = "nonneg", zero_threshold: float = 0.01,
               nnls_iters: int = 50) -> torch.Tensor:
        """
        x: (N, input_dim) -> dense (N, n_codewords) code, nonzero only at
        each row's `n_neighbors` nearest-codeword indices. Kept dense here
        since N is one mini-batch; callers sparsify the accumulated result.
        """
        dists = torch.cdist(x, self.codebook)  # (N, M)
        _, nn_idx = torch.topk(dists, n_neighbors, dim=1, largest=False)  # (N, K)
        B_i = self.codebook[nn_idx]  # (N, K, D)

        c_local = _solve_local(x, B_i, constraint, nnls_iters)  # (N, K)
        if zero_threshold:
            c_local = torch.where(
                c_local.abs() < zero_threshold, torch.zeros_like(c_local), c_local
            )

        code = torch.zeros(x.shape[0], self.n_codewords, device=x.device, dtype=x.dtype)
        code.scatter_(1, nn_idx, c_local)
        return code

    def forward(self, x, **encode_kwargs):
        """
        Convenience only, for interface parity with SimpleAutoencoder.forward
        (-> (x_hat, code) tuple). encode()/decode() are the real primitives;
        forward composes them with no joint loss driving both directions the
        way SAE's does. Call under torch.no_grad() -- no gradient is needed
        through the top-k/solve steps.
        """
        code = self.encode(x, **encode_kwargs)
        return self.decode(code), code


def fit_codebook(model: LLCModel, embeddings: np.ndarray, *,
                  sample_size: "int | None" = None,
                  random_state=None,
                  backend: str = "sklearn",
                  kmeans_kwargs: "dict | None" = None,
                  verbose: bool = False) -> dict:
    """
    Fit `model.codebook` via K-means over `embeddings` (optionally subsampled to
    `sample_size` rows). No learned-codebook refinement (the paper's Algorithm 4.1)
    is performed -- out of scope per the recipe.

    `backend` selects the K-means implementation: "sklearn" (CPU, exact, default),
    "cuml" (GPU, exact, optional dependency -- see pyproject.toml's `gpu-kmeans`
    extra), or "torch_minibatch" (GPU-capable, streaming/minibatch, no extra
    dependencies -- the only backend that avoids holding the full sampled matrix in
    device memory at once; see mesoslide.tools.sparse_coding._kmeans_backends for
    why this differs from "cuml"'s memory-bounded-but-still-full-batch algorithm).
    `kmeans_kwargs` is forwarded to whichever backend's underlying constructor.
    """
    random_state = check_random_state(random_state)
    X = embeddings
    if sample_size is not None and sample_size < len(X):
        idx = random_state.choice(len(X), size=sample_size, replace=False)
        X = X[idx]

    if verbose:
        print(f"Fitting K-means codebook ({backend}): {model.n_codewords} codewords over {X.shape[0]} samples")
    result = fit_kmeans_backend(backend, X, model.n_codewords, random_state, kmeans_kwargs)

    with torch.no_grad():
        model.codebook.data = torch.tensor(result["centers"], dtype=torch.float32)

    return {"inertia": result["inertia"], "n_iter": result["n_iter"]}


# -- constrained least-squares solvers ---------------------------------------

def _solve_local(x: torch.Tensor, B_i: torch.Tensor, constraint: str,
                  nnls_iters: int) -> torch.Tensor:
    """x: (N, D), B_i: (N, K, D) -> c: (N, K)."""
    if constraint == "unconstrained":
        return _solve_unconstrained(x, B_i)
    elif constraint == "shift_invariant":
        return _solve_shift_invariant(x, B_i)
    elif constraint == "nonneg":
        return _solve_nnls_batched(x, B_i, iters=nnls_iters)
    elif constraint == "nonneg_shift_invariant":
        return _solve_nnls_shift_invariant_batched(x, B_i, iters=nnls_iters)
    else:
        raise ValueError(
            f"Unknown constraint {constraint!r}; expected one of "
            "'unconstrained', 'shift_invariant', 'nonneg', 'nonneg_shift_invariant'."
        )


def _gram_and_rhs(x: torch.Tensor, B_i: torch.Tensor):
    """G = B_i B_i^T (N, K, K); rhs = B_i x (N, K)."""
    G = torch.matmul(B_i, B_i.transpose(-1, -2))
    rhs = torch.matmul(B_i, x.unsqueeze(-1)).squeeze(-1)
    return G, rhs


def _solve_unconstrained(x: torch.Tensor, B_i: torch.Tensor) -> torch.Tensor:
    """Closed form: normal equations (B_i B_i^T) c = B_i x, one batched solve."""
    G, rhs = _gram_and_rhs(x, B_i)
    K = G.shape[-1]
    eye = torch.eye(K, device=x.device, dtype=x.dtype)
    G = G + 1e-6 * eye
    return torch.linalg.solve(G, rhs.unsqueeze(-1)).squeeze(-1)


def _solve_shift_invariant(x: torch.Tensor, B_i: torch.Tensor) -> torch.Tensor:
    """
    Closed form (paper Eq. 5-6, specialized to the K-NN local basis, no
    locality-adaptor term). Under the constraint sum(c) = 1,
    ||x - B_i^T c||^2 == ||sum_j c_j (b_j - x)||^2, so the local basis must
    be centered on the query point first: C_i = (B_i - x)(B_i - x)^T. Solving
    with the plain (uncentered) Gram matrix B_i B_i^T -- as one might expect
    by analogy with the unconstrained case -- silently drops x's contribution
    and gives the wrong answer.
    """
    B_centered = B_i - x.unsqueeze(1)  # (N, K, D)
    C = torch.matmul(B_centered, B_centered.transpose(-1, -2))  # (N, K, K)
    N, K, _ = C.shape
    eye = torch.eye(K, device=x.device, dtype=x.dtype)
    C = C + 1e-6 * eye
    ones = torch.ones(N, K, 1, device=x.device, dtype=x.dtype)
    c_tilde = torch.linalg.solve(C, ones).squeeze(-1)
    return c_tilde / c_tilde.sum(dim=-1, keepdim=True)


def _solve_nnls_batched(x: torch.Tensor, B_i: torch.Tensor, iters: int = 50) -> torch.Tensor:
    """
    Batched projected-gradient descent for min 0.5 c^T G c - rhs^T c s.t. c >= 0.

    K is tiny (default 5), so a fixed, modest iteration count converges well.
    Every operation is a batched (N,K,K)@(N,K,1) matmul plus an elementwise
    clamp -- vectorized over the whole mini-batch, unlike a per-sample
    scipy.optimize.nnls loop (which would dominate at mesoslide's scale of
    potentially millions of tiles per cohort).
    """
    G, rhs = _gram_and_rhs(x, B_i)
    N, K, _ = G.shape
    # Cheap Lipschitz proxy (trace >= largest eigenvalue for PSD G) avoids a
    # per-sample eigendecomposition; safe (if slightly conservative) step size.
    L = G.diagonal(dim1=-2, dim2=-1).sum(-1) + 1e-6
    step = (1.0 / L).unsqueeze(-1)

    c = torch.zeros(N, K, device=x.device, dtype=x.dtype)
    for _ in range(iters):
        grad = torch.matmul(G, c.unsqueeze(-1)).squeeze(-1) - rhs
        c = torch.clamp(c - step * grad, min=0.0)
    return c


def _project_simplex_batched(v: torch.Tensor) -> torch.Tensor:
    """
    Euclidean projection of each row of `v` (N, K) onto the probability
    simplex {c : c >= 0, sum(c) = 1}, via sorted cumulative-sum thresholding
    (Held, Wolfe & Crowder 1974 / Duchi et al. 2008). Fully vectorized over
    rows with torch.sort + torch.cumsum -- no per-row Python loop.
    """
    N, K = v.shape
    u, _ = torch.sort(v, dim=-1, descending=True)
    css = torch.cumsum(u, dim=-1)
    idx = torch.arange(1, K + 1, device=v.device, dtype=v.dtype)
    cond = u - (css - 1.0) / idx > 0
    rho = cond.sum(dim=-1, keepdim=True).clamp(min=1)  # (N, 1), number of active coords
    theta = (torch.gather(css, 1, rho - 1) - 1.0) / rho
    return torch.clamp(v - theta, min=0.0)


def _solve_nnls_shift_invariant_batched(x: torch.Tensor, B_i: torch.Tensor,
                                         iters: int = 50) -> torch.Tensor:
    """
    Batched projected-gradient descent onto the simplex for
    min 0.5 c^T G c - rhs^T c  s.t. c >= 0, sum(c) = 1.

    Note (paper's own ablation, Wang et al. CVPR 2010, Fig. 7): this
    constraint mode reconstructs noticeably worse than the other three
    (~62% vs ~73% downstream accuracy on their Caltech-101 benchmark) because
    the reachable set shrinks from an unbounded affine hull to the small
    bounded convex hull of the K neighbors. Implemented here for API
    completeness/symmetry, not recommended as a default.
    """
    G, rhs = _gram_and_rhs(x, B_i)
    N, K, _ = G.shape
    L = G.diagonal(dim1=-2, dim2=-1).sum(-1) + 1e-6
    step = (1.0 / L).unsqueeze(-1)

    c = torch.full((N, K), 1.0 / K, device=x.device, dtype=x.dtype)
    for _ in range(iters):
        grad = torch.matmul(G, c.unsqueeze(-1)).squeeze(-1) - rhs
        c = _project_simplex_batched(c - step * grad)
    return c


def _solve_nnls_scipy_loop(x: np.ndarray, B_i: np.ndarray) -> np.ndarray:
    """
    Reference/oracle implementation only (per-sample scipy.optimize.nnls) --
    never called from LLCModel.encode. Used by tests to validate
    _solve_nnls_batched's accuracy on small random problems.
    """
    from scipy.optimize import nnls

    N, K, D = B_i.shape
    out = np.zeros((N, K))
    for i in range(N):
        c, _ = nnls(B_i[i].T, x[i])
        out[i] = c
    return out


class LocalityConstrainedCoding(TransformerMixin, BaseEstimator):
    """
    Locality-constrained Linear Coding (Wang et al., CVPR 2010): a K-means
    codebook plus a non-parametric per-descriptor encoder (K-NN search +
    constrained least squares). Mirrors SparseAutoencoder's sklearn-style
    fit/transform API and constructor-kwargs-as-hyperparameters convention.
    """

    def __init__(self,
                 n_codewords: int = 2048,
                 n_neighbors: int = 5,
                 constraint: str = "nonneg",
                 zero_threshold: float = 0.01,
                 nnls_iters: int = 50,
                 batch_size: int = 2048,
                 kmeans_backend: str = "sklearn",
                 kmeans_kwargs: "dict | None" = None,
                 random_state=None):
        self.n_codewords = n_codewords
        self.n_neighbors = n_neighbors
        self.constraint = constraint
        self.zero_threshold = zero_threshold
        self.nnls_iters = nnls_iters
        self.batch_size = batch_size
        self.kmeans_backend = kmeans_backend
        self.kmeans_kwargs = kmeans_kwargs
        self.random_state = random_state

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

        self.random_state_ = check_random_state(self.random_state)
        X = validate_data(self, X, accept_sparse=False)
        assert len(X.shape) == 2  # expect X shape = B x d_emb
        if not 0 < fraction <= 1:
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")
        if fraction < 1:
            n_samples = round(fraction * len(X))
            idx = self.random_state_.choice(len(X), size=n_samples, replace=False)
            X = X[idx]

        self.embed_dim_ = self.n_codewords
        self.model_ = LLCModel(input_dim=X.shape[1], n_codewords=self.n_codewords)
        self._training_log = fit_codebook(
            self.model_, X,
            sample_size=sample_size,
            random_state=self.random_state_,
            backend=self.kmeans_backend,
            kmeans_kwargs=self.kmeans_kwargs,
            verbose=verbose,
        )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_.to(device)
        return self

    def transform(self, X, column_keep_indices=None, device=None, *,
                  obsm_key=None, tile_key="tiles", sparse_key_added=None,
                  overwrite=False, save=True):
        if obsm_key is not None:
            from mesoslide.tools._feature_extraction import _write_sparse_features

            slides = X if isinstance(X, (list, tuple)) else [X]
            table_key = f"{tile_key}_table"
            sparse_key = sparse_key_added or f"{obsm_key}_llc"
            for slide in slides:
                table = slide.tables[table_key]
                if overwrite or not any(
                    v.startswith(f"{sparse_key}_") for v in table.var_names
                ):
                    matrix = self.transform(
                        table.obsm[obsm_key], column_keep_indices, device
                    )
                    table = _write_sparse_features(table, sparse_key, matrix)
                    slide.tables[table_key] = table
                    if save:
                        slide.write_element(table_key)
            return X

        check_is_fitted(self)
        X = validate_data(self, X, accept_sparse=False, reset=False)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_.to(device)

        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        rows, cols, vals = [], [], []
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(dataloader)):
                code = self.model_.encode(
                    batch[0].to(device),
                    n_neighbors=self.n_neighbors,
                    constraint=self.constraint,
                    zero_threshold=self.zero_threshold,
                    nnls_iters=self.nnls_iters,
                )
                if column_keep_indices is not None:
                    code = code[:, column_keep_indices]
                arr = code.to_sparse().cpu()
                row_ind, col_ind = arr.indices().numpy()
                value = arr.values().numpy()
                rows.append(row_ind + idx * self.batch_size)
                cols.append(col_ind)
                vals.append(value)

        data = np.concatenate(vals)
        row_ind, col_ind = np.concatenate(rows), np.concatenate(cols)
        n_features = self.embed_dim_ if column_keep_indices is None else len(column_keep_indices)
        return sp.csr_matrix((data, (row_ind, col_ind)), shape=(X.shape[0], n_features))

    def inverse_transform(self, Xt) -> np.ndarray:
        """Reconstruct descriptors from codes via the codebook (x_hat = codes @ B)."""
        check_is_fitted(self)
        dense = Xt.toarray() if sp.issparse(Xt) else np.asarray(Xt)
        # self.model_ may have been left on another device by a prior
        # transform() call, since that moves the model to run inference.
        device = self.model_.codebook.device
        codes = torch.tensor(dense, dtype=torch.float32, device=device)
        with torch.no_grad():
            return self.model_.decode(codes).cpu().numpy()
