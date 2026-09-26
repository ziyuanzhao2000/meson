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
    """

    def __init__(self, d_model, n_latents, threshold=1e-2, normalize=True):
        super().__init__()
        self.d_model = d_model
        self.n_latents = n_latents
        self.threshold = threshold
        self.W = nn.Parameter(torch.randn(n_latents, d_model))
        if normalize:
            self.W.data = F.normalize(self.W.data, p=2, dim=1)

    @property
    def device(self):
        return self.W.device

    def _pursuit(self, x):
        """Greedy matching pursuit; returns codes z of shape (batch, n_latents)."""
        residual = x.clone()
        batch_size = x.shape[0]

        z = torch.zeros(batch_size, self.n_latents, device=x.device, dtype=x.dtype)
        prev_support = torch.zeros_like(z).bool()
        done = torch.zeros(batch_size, dtype=torch.bool, device=x.device)

        while not done.all():
            WTr = torch.relu(residual @ self.W.T)
            values, indices = torch.max(WTr, dim=1, keepdim=True)

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

        return z

    @torch.no_grad()
    def encode(self, x):
        """Sparse codes without gradient tracking."""
        return self._pursuit(x)

    def forward(self, x):
        z = self._pursuit(x)
        return z @ self.W, z


def train_mp_sae(model, embeddings, device="cpu",
                 batch_size=32,
                 num_steps=1000,
                 learning_rate=1e-3,
                 grad_clip_norm=1.0,
                 generator=None,
                 verbose=0):
    """
    Train an MP-SAE on raw (unscaled) embeddings.

    Each step draws `batch_size` rows uniformly with replacement and minimizes
    the per-sample summed squared reconstruction error. Optimizer and schedule
    follow the original implementation: Adam(betas=(0.5, 0.9375)) with a WSD
    schedule (100 warmup steps, 20% cooldown to 0.1x).

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
    expansion_factor : int, default=8
        Dictionary size is `expansion_factor * n_features_in_`.
    threshold : float, default=1e-2
        Residual-norm stopping threshold of the pursuit.
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

    def __init__(self,
                 expansion_factor: int = 8,
                 threshold: float = 1e-2,
                 batch_size: int = 32,
                 num_steps: int = 1000,
                 learning_rate: float = 1e-3,
                 grad_clip_norm: "float | None" = 1.0,
                 normalize_init: bool = True,
                 transform_batch_size: int = 2048,
                 random_state=None):
        self.expansion_factor = expansion_factor
        self.threshold = threshold
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
            verbose: "int | bool" = False):
        if obsm_key is not None:
            from mesoslide._slides import SlideSource
            source = SlideSource(X, tile_key=tile_key)
            X = np.vstack([table.obsm[obsm_key] for _, table in source])

        self.random_state_ = check_random_state(self.random_state)
        X = validate_data(self, X, accept_sparse=False, dtype=np.float32)
        self.embed_dim_ = X.shape[1] * self.expansion_factor

        seed = int(self.random_state_.randint(0, 2**32 - 1))
        torch.manual_seed(seed)
        self.model_ = MatchingPursuitDictionary(
            d_model=X.shape[1],
            n_latents=self.embed_dim_,
            threshold=self.threshold,
            normalize=self.normalize_init,
        )
        generator = torch.Generator().manual_seed(seed)
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._training_log = train_mp_sae(
            self.model_, X,
            device=device,
            batch_size=self.batch_size,
            num_steps=self.num_steps,
            learning_rate=self.learning_rate,
            grad_clip_norm=self.grad_clip_norm,
            generator=generator,
            verbose=verbose,
        )
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
