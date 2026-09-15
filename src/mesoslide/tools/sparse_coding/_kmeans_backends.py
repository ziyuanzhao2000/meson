"""K-means backends for fitting an LLC codebook (see fit_kmeans_backend / _llc.fit_codebook).

Three backends:
  - "sklearn"         CPU, exact Lloyd's K-means (sklearn.cluster.KMeans). Default;
                       unchanged from LLC's original behavior.
  - "cuml"             GPU, exact Lloyd's K-means (cuml.cluster.KMeans). Optional
                       dependency (`uv sync --extra gpu-kmeans`). Bounds peak memory
                       for the pairwise-distance step internally
                       (`max_samples_per_batch`), but still requires a full pass over
                       the entire dataset per iteration -- this is NOT minibatch
                       K-means, just a memory-bounded implementation of the same exact
                       algorithm. See the module docstring on TorchMiniBatchKMeans for
                       the actual difference.
  - "torch_minibatch"  GPU-capable, streaming minibatch K-means (Sculley 2010,
                       TorchMiniBatchKMeans below). Zero new dependencies. Never needs
                       the full dataset resident in memory at once -- this is the
                       actual memory-reduction lever, not cuml's internal batching.
"""

import numpy as np
import torch
from sklearn.utils import check_random_state


def _kmeans_sklearn(X: np.ndarray, n_clusters: int, random_state, kmeans_kwargs: dict) -> dict:
    from sklearn.cluster import KMeans

    kmeans_kwargs = dict(kmeans_kwargs or {})
    kmeans_kwargs.setdefault("n_init", "auto")
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state.randint(0, 2**32 - 1),
        **kmeans_kwargs,
    )
    kmeans.fit(X)
    return {
        "centers": kmeans.cluster_centers_,
        "inertia": kmeans.inertia_,
        "n_iter": kmeans.n_iter_,
    }


def _kmeans_cuml(X: np.ndarray, n_clusters: int, random_state, kmeans_kwargs: dict) -> dict:
    try:
        import cuml
    except ImportError as e:
        raise ImportError(
            "backend='cuml' requires the optional cuml dependency. Install it with "
            "`uv sync --extra gpu-kmeans` (or `pip install mesoslide[gpu-kmeans]`)."
        ) from e

    kmeans_kwargs = dict(kmeans_kwargs or {})
    kmeans = cuml.cluster.KMeans(
        n_clusters=n_clusters,
        random_state=int(random_state.randint(0, 2**32 - 1)),
        **kmeans_kwargs,
    )
    kmeans.fit(X)
    return {
        "centers": np.asarray(kmeans.cluster_centers_),
        "inertia": float(kmeans.inertia_),
        "n_iter": int(getattr(kmeans, "n_iter_", -1)),
    }


def _kmeanspp_init(X_sample: torch.Tensor, n_clusters: int, rng: np.random.RandomState) -> torch.Tensor:
    """Standard k-means++ seeding (Arthur & Vassilvitskii 2007) over an in-memory
    sample. Each subsequent center is drawn with probability proportional to its
    squared distance to the nearest already-chosen center, which spreads the
    initial centers out and avoids the "two centers land in the same blob, one
    blob never gets one" failure mode of plain uniform random init."""
    n = X_sample.shape[0]
    first_idx = rng.randint(0, n)
    centers = [X_sample[first_idx]]
    closest_dist_sq = torch.sum((X_sample - centers[0]) ** 2, dim=1)
    for _ in range(1, n_clusters):
        total = closest_dist_sq.sum()
        if total <= 0:
            # All remaining points coincide with a chosen center; fall back to
            # uniform choice among the rest rather than dividing by zero.
            next_idx = rng.randint(0, n)
        else:
            probs = (closest_dist_sq / total).cpu().numpy()
            next_idx = rng.choice(n, p=probs)
        centers.append(X_sample[next_idx])
        new_dist_sq = torch.sum((X_sample - X_sample[next_idx]) ** 2, dim=1)
        closest_dist_sq = torch.minimum(closest_dist_sq, new_dist_sq)
    return torch.stack(centers)


def _compute_inertia_chunked(X: np.ndarray, centers: torch.Tensor, device, chunk_size: int) -> float:
    """Mean squared distance to the nearest center, computed in row-chunks so this
    one-time diagnostic pass doesn't itself require the full dataset on `device`."""
    total = 0.0
    n = X.shape[0]
    with torch.no_grad():
        for start in range(0, n, chunk_size):
            xb = torch.tensor(X[start:start + chunk_size], dtype=torch.float32, device=device)
            d = torch.cdist(xb, centers)
            total += (d.min(dim=1).values ** 2).sum().item()
    return total / n


class TorchMiniBatchKMeans:
    """
    Streaming minibatch K-means (Sculley, "Web-Scale K-Means Clustering", WWW 2010),
    implemented in plain PyTorch. GPU-capable, zero new dependencies.

    At each step, `batch_size` rows are drawn at random from `X` and moved to
    `device`; only that minibatch plus the current `(n_clusters, D)` centers tensor
    are ever resident on `device` at once -- the full dataset stays on the host
    (`X` is read directly, in slices, never fully copied to the device). This is the
    actual memory-reduction property that distinguishes this from GPU-accelerated but
    still full-batch algorithms (e.g. cuml.cluster.KMeans, which needs every point for
    every iteration, just with its internal distance computation chunked for memory).

    Update rule
    -----------
    All points in a minibatch are first assigned to their nearest *current* center
    (i.e. assignment uses the centers as they stood before this minibatch's update --
    matching Sculley's algorithm, which caches assignments before updating anything).
    Then, instead of Sculley's literal per-point loop (`v[c] += 1; center[c] +=
    (x - center[c]) / v[c]`, processed one point at a time), this aggregates the
    minibatch's assigned points per cluster once via `index_add_` (sum and count) and
    applies a single update per touched cluster:

        eta          = batch_count / (running_count + batch_count)
        center[c]   <- center[c] * (1 - eta) + batch_mean[c] * eta
        running_count[c] += batch_count[c]

    This is not an approximation of Sculley's per-point loop -- it is mathematically
    the *same* running mean (a running/incremental mean is order-invariant: the mean
    of `running_count` old effective observations at the old center plus `batch_count`
    new points equals `old_center + batch_count * (batch_mean - old_center) /
    (running_count + batch_count)`, regardless of what order those new points are
    folded in). Aggregating first just replaces a per-point Python loop with one
    vectorized `index_add_` call per minibatch.

    Empty clusters
    --------------
    A cluster that never receives a single point across the whole run (`running_count
    == 0`) is a known minibatch K-means failure mode -- it never gets updated and
    stays wherever it was initialized. After training, any such cluster is reseeded to
    the point from the most recent minibatch with the largest distance to its own
    assigned center (a simple, cheap heuristic; sklearn's MiniBatchKMeans uses a
    related idea via `reassignment_ratio` during training rather than only at the end
    -- this implementation only reseeds once, post hoc, which is adequate for LLC's
    codebook-fitting use case but not a full reimplementation of sklearn's ongoing
    reassignment logic).

    Parameters
    ----------
    n_clusters : int
    batch_size : int, default 4096
    max_iter : int
        Number of minibatch steps (not full dataset passes), default 100.
    tol, n_iter_no_change : float, int
        Early stopping: if the exponential moving average of per-batch inertia
        changes by less than `tol` for `n_iter_no_change` consecutive checks, stop.
    n_init : int, default 1
        Number of independent random-init runs; the one with the lowest final
        `inertia_` (computed over the full dataset, chunked) is kept.
    init_size : int or None
        Size of the sample used for k-means++ seeding (see `_kmeanspp_init`).
        Defaults to `max(3 * batch_size, 3 * n_clusters)`, capped to the dataset
        size, mirroring sklearn.cluster.MiniBatchKMeans's own default -- large
        enough to spread the initial centers well, small enough to keep the
        one-time init cost (and memory) bounded rather than touching all of `X`.
    random_state : int, RandomState, or None
    device : str or None
        Defaults to "cuda" if available, else "cpu".

    Attributes (after fit)
    -----------------------
    cluster_centers_ : np.ndarray (n_clusters, D)
    inertia_ : float
    n_iter_ : int
    """

    def __init__(self, n_clusters: int, batch_size: int = 4096, max_iter: int = 100,
                 tol: float = 1e-4, n_iter_no_change: int = 10, n_init: int = 1,
                 init_size: "int | None" = None,
                 random_state=None, device=None):
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.tol = tol
        self.n_iter_no_change = n_iter_no_change
        self.n_init = n_init
        self.init_size = init_size
        self.random_state = random_state
        self.device = device

    def fit(self, X: np.ndarray) -> "TorchMiniBatchKMeans":
        device = torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        rng = check_random_state(self.random_state)

        best = None
        for _ in range(self.n_init):
            centers, n_iter = self._fit_once(X, device, rng)
            inertia = _compute_inertia_chunked(X, centers, device, self.batch_size)
            if best is None or inertia < best[2]:
                best = (centers, n_iter, inertia)

        centers, n_iter, inertia = best
        self.cluster_centers_ = centers.detach().cpu().numpy()
        self.inertia_ = inertia
        self.n_iter_ = n_iter
        return self

    def _fit_once(self, X: np.ndarray, device: torch.device, rng: np.random.RandomState):
        n, d = X.shape
        init_size = self.init_size or max(3 * self.batch_size, 3 * self.n_clusters)
        init_size = min(init_size, n)
        init_idx = rng.choice(n, size=init_size, replace=False)
        X_init = torch.tensor(X[init_idx], dtype=torch.float32, device=device)
        centers = _kmeanspp_init(X_init, self.n_clusters, rng)
        running_count = torch.zeros(self.n_clusters, device=device, dtype=torch.float32)

        ema_inertia = None
        prev_ema = None
        stall = 0
        last_batch_x = last_batch_dists = last_batch_assign = None
        step = 0

        for step in range(self.max_iter):
            idx = rng.randint(0, n, size=self.batch_size)
            x_batch = torch.tensor(X[idx], dtype=torch.float32, device=device)

            dists = torch.cdist(x_batch, centers)
            assign = dists.argmin(dim=1)
            min_dists = dists.gather(1, assign.unsqueeze(1)).squeeze(1)
            batch_inertia = (min_dists ** 2).mean().item()

            batch_sum = torch.zeros(self.n_clusters, d, device=device).index_add_(
                0, assign, x_batch
            )
            batch_count = torch.zeros(self.n_clusters, device=device).index_add_(
                0, assign, torch.ones_like(assign, dtype=torch.float32)
            )
            mask = batch_count > 0
            batch_mean = batch_sum[mask] / batch_count[mask].unsqueeze(1)
            eta = (batch_count[mask] / (running_count[mask] + batch_count[mask])).unsqueeze(1)
            centers[mask] = centers[mask] * (1 - eta) + batch_mean * eta
            running_count[mask] += batch_count[mask]

            last_batch_x, last_batch_dists, last_batch_assign = x_batch, dists, assign

            ema_inertia = batch_inertia if ema_inertia is None else 0.9 * ema_inertia + 0.1 * batch_inertia
            if prev_ema is not None and abs(prev_ema - ema_inertia) < self.tol:
                stall += 1
                if stall >= self.n_iter_no_change:
                    break
            else:
                stall = 0
            prev_ema = ema_inertia

        empty = (running_count == 0).nonzero(as_tuple=True)[0]
        if len(empty) > 0 and last_batch_dists is not None:
            own_dist = last_batch_dists.gather(1, last_batch_assign.unsqueeze(1)).squeeze(1)
            order = torch.argsort(own_dist, descending=True)
            for i, c in enumerate(empty.tolist()):
                if i < len(order):
                    centers[c] = last_batch_x[order[i]]

        return centers, step + 1


def _kmeans_torch_minibatch(X: np.ndarray, n_clusters: int, random_state, kmeans_kwargs: dict) -> dict:
    kmeans_kwargs = dict(kmeans_kwargs or {})
    seed = random_state.randint(0, 2**32 - 1)
    model = TorchMiniBatchKMeans(n_clusters=n_clusters, random_state=seed, **kmeans_kwargs)
    model.fit(X)
    return {"centers": model.cluster_centers_, "inertia": model.inertia_, "n_iter": model.n_iter_}


_BACKENDS = {
    "sklearn": _kmeans_sklearn,
    "cuml": _kmeans_cuml,
    "torch_minibatch": _kmeans_torch_minibatch,
}


def fit_kmeans_backend(backend: str, X: np.ndarray, n_clusters: int, random_state, kmeans_kwargs: dict) -> dict:
    if backend not in _BACKENDS:
        raise ValueError(
            f"Unknown kmeans backend {backend!r}; expected one of {sorted(_BACKENDS)}."
        )
    return _BACKENDS[backend](X, n_clusters, random_state, kmeans_kwargs)
