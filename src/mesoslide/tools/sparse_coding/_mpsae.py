"""Matching pursuit sparse autoencoder (MP-SAE).

Encoder is greedy matching pursuit over a learned dictionary `W`
(n_latents x d_model): at each step, pick the atom with the largest positive
correlation with the residual, add that correlation to its code, and subtract
its contribution from the residual. A sample stops when its support stops
changing or its residual norm falls below `threshold`. The decoder is `z @ W`.
Training backpropagates through the pursuit.

Ported from analysis_meson/TB/3D_HnE/notebooks/mpsae.py (MatchingPursuitSAE).
"""
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted, validate_data
from tqdm.auto import tqdm


def get_wsd_scheduler(
    optimizer, n_steps, end_lr_factor=0.1, n_warmup_steps=None, percent_cooldown=0.1
):
    """
    Warmup-stable-decay schedule. See
    https://www.lighton.ai/lighton-blogs/passing-the-torch-training-a-mamba-model-for-smooth-handover
    """
    if n_warmup_steps is None:
        n_warmup_steps = 0.05 * n_steps

    def lr_lambda(step):
        if step < n_warmup_steps:
            return step / n_warmup_steps
        elif step < (1 - percent_cooldown) * n_steps:
            return 1
        else:
            return 1 - (1 - end_lr_factor) * min(
                (step - (1 - percent_cooldown) * n_steps),
                (1 - percent_cooldown) * n_steps,
            ) / (percent_cooldown * n_steps + 1e-2)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


class MatchingPursuitDictionary(nn.Module):
    """Dictionary `W` (n_latents x d_model) with a matching pursuit encoder.

    Unlike the original MatchingPursuitSAE, the optimizer and scheduler are
    not stored on the module; training state lives in `train_mp_sae`.

    max_iter caps the number of pursuit iterations (and so the number of
    active atoms per sample); None runs to convergence, as the original.
    """

    max_iter = None  # class default keeps models pickled before this option loadable

    def __init__(self, d_model, n_latents, threshold=1e-2, normalize=True, max_iter=None):
        super().__init__()
        self.d_model = d_model
        self.n_latents = n_latents
        self.threshold = threshold
        self.max_iter = max_iter
        self.W = nn.Parameter(torch.randn(n_latents, d_model))
        if normalize:
            self.W.data = F.normalize(self.W.data, p=2, dim=1)

    @property
    def device(self):
        return self.W.device

    def _pursuit(self, x, record=False):
        """Greedy matching pursuit; returns codes z of shape (batch, n_latents).

        With record=True, also returns the per-iteration (atom indices, active
        mask) needed by `_replay`.
        """
        residual = x.clone()
        batch_size = x.shape[0]

        z = torch.zeros(batch_size, self.n_latents, device=x.device, dtype=x.dtype)
        prev_support = torch.zeros_like(z).bool()
        done = torch.zeros(batch_size, dtype=torch.bool, device=x.device)
        trace = []
        n_iter = 0

        while not done.all() and (self.max_iter is None or n_iter < self.max_iter):
            n_iter += 1
            WTr = torch.relu(residual @ self.W.T)
            values, indices = torch.max(WTr, dim=1, keepdim=True)
            if record:
                trace.append((indices.squeeze(1), ~done))

            z_ = torch.zeros_like(z)
            z_.scatter_(1, indices, values)
            z = torch.where(done.unsqueeze(1), z, z + z_)

            update = torch.matmul(z_, self.W)
            residual = torch.where(done.unsqueeze(1), residual, residual - update)

            support = z != 0
            # Converged: support unchanged since last step, or residual small enough
            converged = (support == prev_support).all(dim=1) | (
                residual.norm(dim=1) < self.threshold
            )
            done = done | converged
            prev_support = support

        return (z, trace) if record else z

    def _replay(self, x, trace):
        """Differentiable reconstruction from a recorded pursuit.

        Recomputes each selected coefficient relu(<residual, W[i]>) and the
        residual update with gradients, touching only the selected atoms. Same
        function of W as `_pursuit` followed by z @ W, since only the argmax
        entry of each iteration carries gradient; stores (batch, d_model)
        instead of (batch, n_latents) tensors per iteration.
        """
        residual = x
        x_hat = torch.zeros_like(x)
        for indices, active in trace:
            w = self.W[indices]
            v = torch.relu((residual * w).sum(dim=1)) * active
            step = v.unsqueeze(1) * w
            residual = residual - step
            x_hat = x_hat + step
        return x_hat

    @torch.no_grad()
    def encode(self, x):
        """Sparse codes without gradient tracking."""
        return self._pursuit(x)

    def forward(self, x):
        if not torch.is_grad_enabled():
            z = self._pursuit(x)
            return z @ self.W, z
        with torch.no_grad():
            z, trace = self._pursuit(x, record=True)
        return self._replay(x, trace), z


def train_mp_sae(model, embeddings, device="cpu",
                 batch_size=32,
                 num_steps=1000,
                 learning_rate=1e-3,
                 grad_clip_norm=1.0,
                 generator=None,
                 verbose=0,
                 checkpoint_every=None,
                 on_checkpoint=None):
    """
    Train an MP-SAE on raw (unscaled) embeddings.

    Each step draws `batch_size` rows uniformly with replacement and minimizes
    the per-sample summed squared reconstruction error. Optimizer and schedule
    follow the original implementation: Adam(betas=(0.5, 0.9375)) with a WSD
    schedule (100 warmup steps, 20% cooldown to 0.1x).

    If `checkpoint_every` is set, `on_checkpoint(step, mses)` is called every
    `checkpoint_every` steps and after the last step.

    Returns a log dict with per-step `mse`.
    """
    # Rows stay on CPU; only each minibatch is moved to `device`
    X = torch.as_tensor(embeddings, dtype=torch.float32)
    n = X.shape[0]

    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.9375))
    scheduler = get_wsd_scheduler(
        optimizer,
        n_steps=num_steps,
        n_warmup_steps=100,
        percent_cooldown=0.2,
        end_lr_factor=0.1,
    )

    mses = []
    for step in tqdm(range(1, num_steps + 1)):
        idx = torch.randint(0, n, (batch_size,), generator=generator)
        x = X[idx].to(device)

        x_hat, _ = model(x)
        loss = ((x_hat - x) ** 2).sum(dim=-1).mean(dim=-1)

        optimizer.zero_grad()
        loss.backward()
        max_norm = float("inf") if grad_clip_norm is None else grad_clip_norm
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        scheduler.step()

        mses.append(loss.item())
        if verbose and step % verbose == 0:
            print(f"[{step}/{num_steps}] mse={mses[-1]:.6f}")
        if checkpoint_every and (step % checkpoint_every == 0 or step == num_steps):
            on_checkpoint(step, mses)

    model.eval()
    return {"mse": mses}


class MatchingPursuitSAE(TransformerMixin, BaseEstimator):
    """
    Matching pursuit sparse autoencoder with the sparse-coding fit/transform API
    shared by SparseAutoencoder, LocalityConstrainedCoding and
    MiniBatchDictionaryCoding.

    Inputs are used as given (no scaling), matching the original MP-SAE.

    Parameters
    ----------
    expansion_factor : float, default=8
        Dictionary size is `round(expansion_factor * n_features_in_)`; values
        below 1 give an undercomplete dictionary (e.g. 0.25 -> 256 atoms for
        1,024-dim inputs).
    threshold : float, default=1e-2
        Residual-norm stopping threshold of the pursuit.
    max_iter : int or None, default=None
        Maximum pursuit iterations, which bounds the number of active features
        per patch. None runs to convergence, as the original MP-SAE.
    batch_size : int, default=32
        Rows per training step.
    num_steps : int, default=1000
        Training steps.
    learning_rate : float, default=1e-3
    grad_clip_norm : float or None, default=1.0
    normalize_init : bool, default=True
        Initialize dictionary rows to unit norm.
    transform_batch_size : int, default=2048
        Rows encoded per block in transform.
    random_state : int, RandomState or None
    """

    max_iter = None  # class default keeps models pickled before this option loadable

    def __init__(self,
                 expansion_factor: float = 8,
                 threshold: float = 1e-2,
                 max_iter: "int | None" = None,
                 batch_size: int = 32,
                 num_steps: int = 1000,
                 learning_rate: float = 1e-3,
                 grad_clip_norm: "float | None" = 1.0,
                 normalize_init: bool = True,
                 transform_batch_size: int = 2048,
                 random_state=None):
        self.expansion_factor = expansion_factor
        self.threshold = threshold
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.learning_rate = learning_rate
        self.grad_clip_norm = grad_clip_norm
        self.normalize_init = normalize_init
        self.transform_batch_size = transform_batch_size
        self.random_state = random_state

    def fit(self, X, y=None, *,
            obsm_key=None, tile_key="tiles",
            device=None,
            verbose: "int | bool" = False,
            checkpoint_every: "int | None" = None,
            checkpoint_dir=None):
        """
        Train the dictionary.

        checkpoint_every, checkpoint_dir: if both set, save a loadable copy of
        the estimator (CPU weights, MSE log so far) as
        `checkpoint_dir/step_<step>.joblib` every `checkpoint_every` steps
        and after the last step.
        """
        if (checkpoint_every is None) != (checkpoint_dir is None):
            raise ValueError("checkpoint_every and checkpoint_dir must be set together")
        if obsm_key is not None:
            from mesoslide._slides import SlideSource
            source = SlideSource(X, tile_key=tile_key)
            X = np.vstack([table.obsm[obsm_key] for _, table in source])

        self.random_state_ = check_random_state(self.random_state)
        X = validate_data(self, X, accept_sparse=False, dtype=np.float32)
        self.embed_dim_ = int(round(X.shape[1] * self.expansion_factor))

        seed = int(self.random_state_.randint(0, 2**32 - 1))
        torch.manual_seed(seed)
        self.model_ = MatchingPursuitDictionary(
            d_model=X.shape[1],
            n_latents=self.embed_dim_,
            threshold=self.threshold,
            normalize=self.normalize_init,
            max_iter=self.max_iter,
        )
        generator = torch.Generator().manual_seed(seed)
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        on_checkpoint = None
        if checkpoint_every is not None:
            import copy
            from pathlib import Path
            import joblib

            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            def on_checkpoint(step, mses):
                snapshot = copy.copy(self)
                snapshot.model_ = copy.deepcopy(self.model_).cpu().eval()
                snapshot._training_log = {"mse": list(mses)}
                snapshot.checkpoint_step_ = step
                joblib.dump(snapshot, checkpoint_dir / f"step_{step:05d}.joblib")

        self._training_log = train_mp_sae(
            self.model_, X,
            device=device,
            batch_size=self.batch_size,
            num_steps=self.num_steps,
            learning_rate=self.learning_rate,
            grad_clip_norm=self.grad_clip_norm,
            generator=generator,
            verbose=verbose,
            checkpoint_every=checkpoint_every,
            on_checkpoint=on_checkpoint,
        )
        # CPU weights so the saved model loads without a GPU; transform moves it to its device
        self.model_.cpu()
        return self

    @property
    def components_(self):
        """Dictionary atoms, shape (embed_dim_, n_features_in_)."""
        check_is_fitted(self)
        return self.model_.W.detach().cpu().numpy()

    def transform(self, X, column_keep_indices=None, device=None, *,
                  obsm_key=None, tile_key="tiles", sparse_key_added=None,
                  overwrite=False, save=True, progress_bar=True):
        if obsm_key is not None:
            from mesoslide.tools._feature_extraction import _write_sparse_features

            slides = X if isinstance(X, (list, tuple)) else [X]
            table_key = f"{tile_key}_table"
            sparse_key = sparse_key_added or f"{obsm_key}_mpsae"
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
        X = validate_data(self, X, accept_sparse=False, reset=False, dtype=np.float32)
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_.to(device)
        self.model_.eval()

        bs = self.transform_batch_size
        rows, cols, vals = [], [], []
        starts = range(0, X.shape[0], bs)
        for start in tqdm(starts, disable=not progress_bar):
            batch = torch.as_tensor(X[start:start + bs]).to(device)
            z = self.model_.encode(batch)
            if column_keep_indices is not None:
                z = z[:, column_keep_indices]
            row_ind, col_ind = z.nonzero(as_tuple=True)
            vals.append(z[row_ind, col_ind].cpu().numpy())
            rows.append(row_ind.cpu().numpy() + start)
            cols.append(col_ind.cpu().numpy())

        n_features = self.embed_dim_ if column_keep_indices is None else len(column_keep_indices)
        if not rows:
            return sp.csr_matrix((X.shape[0], n_features), dtype=np.float32)
        return sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(X.shape[0], n_features),
        )
