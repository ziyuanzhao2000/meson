import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from torch.optim import Adam
from tqdm.auto import tqdm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data
from sklearn.utils import check_random_state
import numpy as np
import scipy.sparse as sp


class SimpleAutoencoder(nn.Module):
    def __init__(self, input_dim, expansion_factor=64):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = int(input_dim * expansion_factor)

        # Simple encoder and decoder
        self.encoder = nn.Linear(input_dim, self.hidden_dim, bias=True)
        self.decoder = nn.Linear(self.hidden_dim, input_dim, bias=True)

        self._initialize_weights()

    def _initialize_weights(self):
        # based on the notes here: https://transformer-circuits.pub/2024/april-update/index.html#training-saes
        with torch.no_grad():
            # Initialize encoder weights randomly
            encoder_weights = torch.randn(self.hidden_dim, self.input_dim)
            # Normalize columns to have random L2 norms between 0.05 and 1
            # Using 0.1 as suggested in the blog
            norms = torch.norm(encoder_weights, dim=1, keepdim=True)
            encoder_weights = encoder_weights / norms * 0.1
            self.encoder.weight.data = encoder_weights

            # Initialize decoder as transpose of encoder
            self.decoder.weight.data = self.encoder.weight.data.t().clone() # materialize to avoid shared storage

            # Initialize biases to zero
            self.encoder.bias.data.zero_()
            self.decoder.bias.data.zero_()

    def encode(self, x):
        """Sparse codes: encoder with ReLU activation."""
        return F.relu(self.encoder(x))

    def forward(self, x):
        h = self.encode(x)
        x_hat = self.decoder(h)
        return x_hat, h

def train_simple_sae(model, embeddings, device='cpu',
              batch_size=2048,
              num_steps=200000,
              min_lambda=1e-6,
              max_lambda=1,
              target_sparsity=0.001,
            #   epsilon=0.05,
              learning_rate=5e-5,
              verbose=100,
              input_norm="sqrt_d",
              lambda_mode="adaptive",
              l1_coefficient=None,
              loss_normalization="legacy"):
    """
    Train the SAE.

    input_norm: "sqrt_d" scales inputs so E[||x||^2] = sqrt(d); "d" so E[||x||^2] = d.
    lambda_mode: "adaptive" tunes lambda in [min_lambda, max_lambda] toward
        target_sparsity; "fixed" uses l1_coefficient for all steps.
    loss_normalization: "legacy" uses MSE averaged over batch and features plus
        an L1 term summed over the batch; "per_sample" sums both over features
        and averages over the batch.
    """
    if input_norm not in ("sqrt_d", "d"):
        raise ValueError(f"input_norm must be 'sqrt_d' or 'd', got {input_norm!r}")
    if lambda_mode not in ("adaptive", "fixed"):
        raise ValueError(f"lambda_mode must be 'adaptive' or 'fixed', got {lambda_mode!r}")
    if lambda_mode == "fixed" and l1_coefficient is None:
        raise ValueError("l1_coefficient is required when lambda_mode='fixed'")
    if loss_normalization not in ("legacy", "per_sample"):
        raise ValueError(
            f"loss_normalization must be 'legacy' or 'per_sample', got {loss_normalization!r}"
        )

    # Scale dataset
    print(embeddings.shape)
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
    n = embeddings_tensor.shape[1]
    current_norm = torch.mean(torch.sum(embeddings_tensor**2, dim=1))
    if input_norm == "sqrt_d":
        target_norm = torch.sqrt(torch.tensor(n, dtype=torch.float32))
    else:
        target_norm = torch.tensor(n, dtype=torch.float32)
    scale_factor = torch.sqrt(target_norm / current_norm)
    embeddings_tensor = embeddings_tensor * scale_factor
    print("Scale factor:", scale_factor)

    dataset = torch.utils.data.TensorDataset(embeddings_tensor)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    model = model.to(device)
    optimizer = Adam(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0
    )

    losses = []
    sparsities = []
    recon_losses = []
    sparsity_losses = []

    step = 0

    current_lambda = l1_coefficient if lambda_mode == "fixed" else min_lambda

    for epoch in tqdm(range((num_steps + len(dataloader) - 1) // len(dataloader))):
        if step >= num_steps:
            break

        for batch in dataloader:
            if step >= num_steps:
                break

            x = batch[0].to(device)

            # Forward pass
            x_hat, h = model(x)

            # Calculate losses
            weighted_l1 = torch.abs(h) * torch.norm(model.decoder.weight, dim=0)
            if loss_normalization == "legacy":
                recon_loss = F.mse_loss(x_hat, x)
                sparsity_loss = current_lambda * torch.sum(weighted_l1)
            else:
                recon_loss = ((x_hat - x) ** 2).sum(dim=1).mean()
                sparsity_loss = current_lambda * weighted_l1.sum(dim=1).mean()

            # Add cosine similarity penalty between dictionary vectors
            # Normalize decoder weights
            # normalized_weights = F.normalize(model.decoder.weight, p=2, dim=0)
            # Calculate cosine similarity matrix
            # cosine_sim = torch.mm(normalized_weights.t(), normalized_weights)
            # Zero out diagonal (self-similarity)
            # cosine_sim.fill_diagonal_(0)
            # Get maximum similarity
            # max_cosine_sim = torch.max(cosine_sim)
            # cosine_penalty = epsilon * max_cosine_sim

            loss = recon_loss + sparsity_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            # The gradient norm is clipped to 1
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Update learning rate
            if step > 0.8 * num_steps:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate * (1 - (step - 0.8 * num_steps) / (0.2 * num_steps))

            # Adaptive lambda adjustment
            # current_sparsity = (h > 0).float().mean().item()
            # if step % 10 == 0:  # Adjust every 10 steps
            #     if current_sparsity < target_sparsity * 0.9:  # Too sparse
            #         current_lambda = current_lambda * 0.95
            #     elif current_sparsity > target_sparsity * 1.1:  # Not sparse enough
            #         current_lambda = current_lambda * 1.05

            current_sparsity = (h > 0).float().mean().item()
            if lambda_mode == "adaptive" and step % 10 == 0:  # Adjust every 10 steps
                if current_sparsity < target_sparsity * 0.8:  # Too sparse
                    current_lambda = max(min_lambda, current_lambda * 0.95)
                elif current_sparsity > target_sparsity * 1.2:  # Not sparse enough
                    current_lambda = min(max_lambda, current_lambda * 1.05)

            # Update lambda linearly
            # if step < 0.05 * num_steps:  # First 5% of steps
            #     current_lambda = min_lambda + (max_lambda - min_lambda) * (step / (0.05 * num_steps))
            # else:
            #     current_lambda = max_lambda

            # Track metrics
            losses.append(loss.item())
            sparsities.append(current_sparsity)
            recon_losses.append(recon_loss.item())
            sparsity_losses.append(sparsity_loss.item())

            if verbose and step % verbose == 0:
                print(f"\nStep {step}, epoch {epoch}")
                print(f"Total Loss: {losses[-1]:.4f}")
                print(f"Recon Loss: {recon_losses[-1]:.4f}")
                print(f"Sparsity Loss: {sparsity_losses[-1]:.4f}")
                # print(f"Cosine Penalty: {cosine_penalty.item():.4f}")  # Added logging
                print(f"Sparsity (L0): {sparsities[-1]:.4f}")
                print(f"Learning rate: {optimizer.param_groups[0]['lr']:.2e}")
                print(f"Lambda: {current_lambda:.2e}")

            step += 1

    logs = {
        'losses': losses,
        'sparsities': sparsities,
        'recon_losses': recon_losses,
        'sparsity_losses': sparsity_losses
    }
    return scale_factor, logs

class SparseAutoencoder(TransformerMixin, BaseEstimator):
    def __init__(self,
                 expansion_factor: int = 64,
                 batch_size: int = 2048,
                 num_steps: int = 200000,
                 min_lambda: float = 1e-6,  # Start very small
                 max_lambda: int = 1,
                 target_sparsity: float = 0.001,
                 learning_rate=5e-5,
                 random_state=None,
                 input_norm: str = "sqrt_d",
                 lambda_mode: str = "adaptive",
                 l1_coefficient: "float | None" = None,
                 loss_normalization: str = "legacy"):
        self.input_norm = input_norm
        self.lambda_mode = lambda_mode
        self.l1_coefficient = l1_coefficient
        self.loss_normalization = loss_normalization
        self.expansion_factor = expansion_factor
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.min_lambda = min_lambda
        self.max_lambda = max_lambda
        self.learning_rate = learning_rate
        self.target_sparsity = target_sparsity
        self.random_state = random_state

    def fit(self, X, y=None, *,
            obsm_key=None, tile_key="tiles",
            device = None,
            fraction: float = 1.0,
            verbose: "int | bool" = False):
        if obsm_key is not None:
            from mesoslide._slides import SlideSource
            source = SlideSource(X, tile_key=tile_key)
            X = np.vstack([table.obsm[obsm_key] for _, table in source])

        self.random_state_ = check_random_state(self.random_state)
        X = validate_data(self, X, accept_sparse=False)
        assert len(X.shape) == 2 # expect X shape = B x d_emb
        if not 0 < fraction <= 1:
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")
        if fraction < 1:
            n_samples = round(fraction * len(X))
            idx = self.random_state_.choice(len(X), size=n_samples, replace=False)
            X = X[idx]
        print(X.shape)
        self.embed_dim_ = X.shape[1] * self.expansion_factor
        torch.manual_seed(int(self.random_state_.randint(0, 2**32 - 1)))
        self.model_ = SimpleAutoencoder(input_dim=X.shape[1],
                                        expansion_factor=self.expansion_factor)
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        self.scale_factor_, self._training_log = train_simple_sae(
            model=self.model_,
            embeddings=X,
            device=device,
            batch_size=self.batch_size,
            num_steps=self.num_steps,
            min_lambda=self.min_lambda,
            max_lambda=self.max_lambda,
            target_sparsity=self.target_sparsity,
            learning_rate=self.learning_rate,
            verbose=verbose,
            input_norm=self.input_norm,
            lambda_mode=self.lambda_mode,
            l1_coefficient=self.l1_coefficient,
            loss_normalization=self.loss_normalization,
        )
        return self

    def transform(self, X, column_keep_indices=None, device=None, *,
                  obsm_key=None, tile_key="tiles", sparse_key_added=None,
                  overwrite=False, save=True, progress_bar=True):
        if obsm_key is not None:
            from mesoslide.tools._feature_extraction import _write_sparse_features

            slides = X if isinstance(X, (list, tuple)) else [X]
            table_key = f"{tile_key}_table"
            sparse_key = sparse_key_added or f"{obsm_key}_sae"
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
        X = validate_data(self, X, accept_sparse=False, reset=False)
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_.to(device)
        # X may be large so we need to use dataloader
        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32) * self.scale_factor_)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        rows = []
        cols = []
        vals = []
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(dataloader, disable=not progress_bar)):
                X_ = self.model_.encode(batch[0].to(device))
                if column_keep_indices is not None:
                    X_ = X_[:, column_keep_indices]
                arr = X_.to_sparse().cpu()
                row_ind, col_ind = arr.indices().numpy()
                value = arr.values().numpy()
                rows.append(row_ind + idx * self.batch_size)
                cols.append(col_ind)
                vals.append(value)
        data = np.concatenate(vals)
        row_ind, col_ind = np.concatenate(rows), np.concatenate(cols)
        n_features = self.embed_dim_ if column_keep_indices is None else len(column_keep_indices)
        X_sparse = sp.csr_matrix((data, (row_ind, col_ind)),
                                    shape=(X.shape[0], n_features))
        return X_sparse

    # def load(self, file_path):
    #     device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #     self.model_.load_state_dict(torch.load(file_path),
    #                                 map_location=torch.device(device))
