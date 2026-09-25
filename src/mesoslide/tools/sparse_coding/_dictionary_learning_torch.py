"""PyTorch backend for MiniBatchDictionaryCoding.

`fista_lasso` solves the same per-row lasso as sklearn's `sparse_encode`
(lasso_lars / lasso_cd), min_c 0.5 ||x - c D||^2 + alpha ||c||_1, optionally
with c >= 0. `fit_dictionary_torch` ports sklearn's
`MiniBatchDictionaryLearning.fit` (online dictionary learning, Mairal et al.
2009) with `fista_lasso` as the sparse coder, so both can run on GPU.
"""

import numpy as np
import torch


def _prox(v: torch.Tensor, thr, positive: bool) -> torch.Tensor:
    """Proximal operator of thr * ||.||_1 (plus the c >= 0 indicator if positive)."""
    if positive:
        return torch.clamp(v - thr, min=0.0)
    return torch.sign(v) * torch.clamp(v.abs() - thr, min=0.0)


def lipschitz_constant(G: torch.Tensor) -> torch.Tensor:
    """Largest eigenvalue of the Gram matrix G = D D^T, i.e. ||D||_2^2."""
    return torch.linalg.eigvalsh(G)[-1].clamp_min(torch.finfo(G.dtype).tiny)


def fista_lasso(X: torch.Tensor, D: torch.Tensor, alpha: float, *,
                positive: bool,
                max_iter: int = 1000,
                tol: float = 1e-4,
                check_every: int = 10,
                G: "torch.Tensor | None" = None,
                L: "torch.Tensor | None" = None) -> torch.Tensor:
    """
    Batched FISTA (Beck & Teboulle 2009) for min_c 0.5 ||x - c D||^2 + alpha ||c||_1
    per row of X, with per-row gradient-based adaptive restart (O'Donoghue &
    Candes 2015).

    Parameters
    ----------
    X : (n, d) tensor
    D : (k, d) tensor, dictionary atoms as rows.
    alpha : float
        L1 penalty; same scale as sklearn's `sparse_encode(alpha=...)`.
    positive : bool
        Constrain codes to be nonnegative.
    max_iter : int
    tol : float
        Stop when the largest entry of the gradient mapping (the KKT residual)
        over the batch is <= tol * alpha. Checked every `check_every` iterations.
    G, L : tensor, optional
        Precomputed D D^T and its largest eigenvalue, reused across batches.

    Returns
    -------
    (n, k) tensor of codes.
    """
    if G is None:
        G = D @ D.T
    if L is None:
        L = lipschitz_constant(G)
    XDt = X @ D.T
    step = 1.0 / L
    thr = alpha / L
    stop = tol * max(float(alpha), torch.finfo(X.dtype).eps)

    c = torch.zeros_like(XDt)
    y = c
    t = torch.ones(X.shape[0], 1, device=X.device, dtype=X.dtype)
    for it in range(1, max_iter + 1):
        c_new = _prox(y - step * (y @ G - XDt), thr, positive)
        # Restart momentum on rows where it points against the proximal step.
        restart = ((y - c_new) * (c_new - c)).sum(dim=1, keepdim=True) > 0
        t = torch.where(restart, torch.ones_like(t), t)
        t_new = (1.0 + torch.sqrt(1.0 + 4.0 * t * t)) / 2.0
        y = c_new + ((t - 1.0) / t_new) * (c_new - c)
        c, t = c_new, t_new
        if it % check_every == 0:
            # Gradient mapping at c; zero iff c is optimal.
            residual = L * (c - _prox(c - step * (c @ G - XDt), thr, positive))
            if residual.abs().max() <= stop:
                break
    return c


def _update_dict_torch(dictionary: torch.Tensor, Y: torch.Tensor,
                       A: torch.Tensor, B: torch.Tensor, *,
                       positive: bool, generator: torch.Generator) -> int:
    """
    In-place port of sklearn.decomposition._dict_learning._update_dict: block
    coordinate descent over atoms using the online statistics A = sum c^T c
    and B = sum x^T c. Atoms with A[k, k] <= 1e-6 are resampled from the batch
    with 1% noise. Returns the number of resampled atoms.
    """
    n_components, n_features = dictionary.shape
    # A is fixed during the sweep, so the unused-atom mask needs one sync only.
    unused = (torch.diagonal(A) <= 1e-6).cpu().numpy()
    n_unused = int(unused.sum())
    replacements = None
    if n_unused:
        idx = torch.randint(Y.shape[0], (n_unused,), generator=generator)
        newd = Y[idx.to(Y.device)]
        noise_level = 0.01 * newd.std(dim=1, unbiased=False, keepdim=True)
        noise_level = torch.where(noise_level > 0, noise_level, 0.01)
        noise = torch.randn(n_unused, n_features, generator=generator, dtype=Y.dtype).to(Y.device)
        replacements = newd + noise * noise_level

    j = 0
    for k in range(n_components):
        if unused[k]:
            dictionary[k] = replacements[j]
            j += 1
        else:
            dictionary[k] += (B[:, k] - A[k] @ dictionary) / A[k, k]
        if positive:
            dictionary[k].clamp_(min=0.0)
        # Projection onto ||d_k|| <= 1.
        dictionary[k] /= torch.linalg.vector_norm(dictionary[k]).clamp_min(1.0)
    return n_unused


def _initial_dictionary(X0: torch.Tensor, n_components: int) -> torch.Tensor:
    """
    Top right-singular vectors of X0 scaled by their singular values, zero-padded
    to n_components rows (sklearn's _initialize_dict, using an exact SVD via the
    d x d Gram matrix instead of randomized SVD).
    """
    gram = (X0.T @ X0).double()
    evals, evecs = torch.linalg.eigh(gram)  # ascending
    evals, evecs = evals.flip(0).clamp_min(0.0), evecs.flip(1)
    rank = min(n_components, X0.shape[0], X0.shape[1])
    dictionary = torch.zeros(n_components, X0.shape[1], dtype=X0.dtype, device=X0.device)
    dictionary[:rank] = (evals[:rank].sqrt()[:, None] * evecs[:, :rank].T).to(X0.dtype)
    return dictionary


def fit_dictionary_torch(X: np.ndarray, *,
                         n_components: int,
                         alpha: float,
                         batch_size: int = 2048,
                         max_iter: int = 50,
                         max_steps: "int | None" = None,
                         tol: float = 1e-3,
                         max_no_improvement: "int | None" = 10,
                         positive_code: bool = True,
                         positive_dict: bool = False,
                         fista_max_iter: int = 1000,
                         fista_tol: float = 1e-4,
                         device: str = "cuda",
                         seed: int = 0,
                         mean: "np.ndarray | None" = None,
                         scale: float = 1.0,
                         dict_init: "np.ndarray | None" = None,
                         shuffle: bool = True,
                         init_sample_size: int = 100_000,
                         verbose: "int | bool" = False) -> "tuple[np.ndarray, dict]":
    """
    Port of sklearn's MiniBatchDictionaryLearning.fit with FISTA sparse coding.

    X stays in host memory; each minibatch is copied to `device`, then scaled
    by `scale` and centered with `mean` there, so the full matrix is never
    duplicated. Differences from
    sklearn: rows are shuffled by index instead of copying X, and the initial
    dictionary is an exact SVD of the first `init_sample_size` shuffled rows
    rather than a randomized SVD of all rows.

    Returns
    -------
    components : (n_components, n_features) ndarray
    log : dict with "n_steps" and "n_iter" (epochs, as sklearn's n_iter_).
    """
    rng = np.random.RandomState(seed)
    generator = torch.Generator().manual_seed(seed)
    n_samples, n_features = X.shape
    dtype = torch.float64 if X.dtype == np.float64 else torch.float32
    batch_size = min(batch_size, n_samples)
    order = rng.permutation(n_samples) if shuffle else None
    mean_t = torch.as_tensor(
        np.zeros(n_features) if mean is None else mean, dtype=dtype, device=device)

    def load(start, stop):
        rows = X[order[start:stop]] if shuffle else X[start:stop]
        return torch.as_tensor(rows, dtype=dtype).to(device, non_blocking=True) * scale - mean_t

    if dict_init is None:
        dictionary = _initial_dictionary(load(0, min(n_samples, init_sample_size)), n_components)
    else:
        dictionary = torch.as_tensor(np.array(dict_init), dtype=dtype, device=device).clone()
    old_dict = dictionary.clone()

    A = torch.zeros(n_components, n_components, dtype=dtype, device=device)
    B = torch.zeros(n_features, n_components, dtype=dtype, device=device)
    ewa_cost, ewa_cost_min, no_improvement = None, None, 0

    n_steps_per_iter = int(np.ceil(n_samples / batch_size))
    n_steps = max_iter * n_steps_per_iter if max_steps is None else max_steps
    step = -1  # allows max_iter = 0
    for step in range(n_steps):
        start = (step % n_steps_per_iter) * batch_size
        Xb = load(start, min(start + batch_size, n_samples))
        bsz = Xb.shape[0]

        # sklearn _minibatch_step
        code = fista_lasso(Xb, dictionary, alpha, positive=positive_code,
                           max_iter=fista_max_iter, tol=fista_tol)
        batch_cost = (0.5 * ((Xb - code @ dictionary) ** 2).sum()
                      + alpha * code.abs().sum()) / bsz

        # sklearn _update_inner_stats
        if step < bsz - 1:
            theta = (step + 1) * bsz
        else:
            theta = bsz ** 2 + step + 1 - bsz
        beta = (theta + 1 - bsz) / (theta + 1)
        A.mul_(beta).add_(code.T @ code / bsz)
        B.mul_(beta).add_(Xb.T @ code / bsz)

        n_unused = _update_dict_torch(dictionary, Xb, A, B,
                                      positive=positive_dict, generator=generator)

        # sklearn _check_convergence
        cost = float(batch_cost)
        if verbose:
            msg = f"Minibatch step {step + 1}/{n_steps}: mean batch cost: {cost}"
            print(msg + (f", {n_unused} unused atoms resampled" if n_unused else ""))
        if step + 1 > min(100, n_samples / bsz):
            if ewa_cost is None:
                ewa_cost = cost
            else:
                w = min(bsz / (n_samples + 1), 1)
                ewa_cost = ewa_cost * (1 - w) + cost * w
            dict_diff = float(torch.linalg.norm(dictionary - old_dict)) / n_components
            if tol > 0 and dict_diff <= tol:
                if verbose:
                    print(f"Converged (small dictionary change) at step {step + 1}/{n_steps}")
                break
            if ewa_cost_min is None or ewa_cost < ewa_cost_min:
                no_improvement, ewa_cost_min = 0, ewa_cost
            else:
                no_improvement += 1
            if max_no_improvement is not None and no_improvement >= max_no_improvement:
                if verbose:
                    print(f"Converged (lack of improvement in objective function) "
                          f"at step {step + 1}/{n_steps}")
                break
        old_dict.copy_(dictionary)

    n_steps_done = step + 1
    log = {"n_steps": n_steps_done, "n_iter": float(np.ceil(n_steps_done / n_steps_per_iter))}
    return dictionary.cpu().numpy(), log
