import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from torch.optim import Adam
from tqdm import tqdm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
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
            self.decoder.weight.data = self.encoder.weight.data.t()
            
            # Initialize biases to zero
            self.encoder.bias.data.zero_()
            self.decoder.bias.data.zero_()
    
    def forward(self, x):
        # Encoder with ReLU activation
        h = F.relu(self.encoder(x))
        # Decoder
        x_hat = self.decoder(h)
        return x_hat, h

def train_simple_sae(model, embeddings, device='cpu', 
              batch_size=2048,
              num_steps=200000,
              initial_lambda=1e-6,  
              final_lambda=1,
              target_sparsity=0.001,
            #   epsilon=0.05,
              learning_rate=5e-5,
              verbose=100):  # Target 1% activation
    """
    Train the SAE with modified sparsity control
    """
    # Scale dataset
    print(embeddings.shape)
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
    n = embeddings_tensor.shape[1]
    current_norm = torch.mean(torch.sum(embeddings_tensor**2, dim=1))
    target_norm = torch.sqrt(torch.tensor(n, dtype=torch.float32))
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
    
    # Adaptive lambda control
    current_lambda = initial_lambda
    
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
            recon_loss = F.mse_loss(x_hat, x)
            sparsity_loss = current_lambda * torch.sum(
                torch.abs(h) * torch.norm(model.decoder.weight, dim=0)
            )
            
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
                    param_group['lr'] = 5e-5 * (1 - (step - 0.8 * num_steps) / (0.2 * num_steps))
            
            # Adaptive lambda adjustment
            # current_sparsity = (h > 0).float().mean().item()
            # if step % 10 == 0:  # Adjust every 10 steps
            #     if current_sparsity < target_sparsity * 0.9:  # Too sparse
            #         current_lambda = current_lambda * 0.95
            #     elif current_sparsity > target_sparsity * 1.1:  # Not sparse enough
            #         current_lambda = current_lambda * 1.05
            
            current_sparsity = (h > 0).float().mean().item()
            if step % 10 == 0:  # Adjust every 10 steps
                if current_sparsity < target_sparsity * 0.8:  # Too sparse
                    current_lambda = max(initial_lambda, current_lambda * 0.95)
                elif current_sparsity > target_sparsity * 1.2:  # Not sparse enough
                    current_lambda = min(final_lambda, current_lambda * 1.05)

            # Update lambda linearly
            # if step < 0.05 * num_steps:  # First 5% of steps
            #     current_lambda = initial_lambda + (final_lambda - initial_lambda) * (step / (0.05 * num_steps))
            # else:
            #     current_lambda = final_lambda
            
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
                 initial_lambda: float = 1e-6,  # Start very small
                 final_lambda: int = 1,
                 target_sparsity: float = 0.001,
                 learning_rate=5e-5,
                 random_state=None):
        self.expansion_factor = expansion_factor
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.initial_lambda = initial_lambda
        self.final_lambda = final_lambda
        # self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.target_sparsity = target_sparsity
        self.random_state = random_state

    def fit(self, X, y=None, 
            verbose: int | bool = False):
        self.random_state_ = check_random_state(self.random_state)
        X = self._validate_data(X, accept_sparse=False)
        assert len(X.shape) == 2 # expect X shape = B x d_emb 
        self.embed_dim_ = X.shape[1] * self.expansion_factor
        self.model_ = SimpleAutoencoder(input_dim=X.shape[1], 
                                        expansion_factor=self.expansion_factor)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        torch.manual_seed(self.random_state_.random(1))
        self.scale_factor_, self._training_log = train_simple_sae(
            model=self.model_,
            embeddings=X,
            device=device,
            batch_size=self.batch_size,
            num_steps=self.num_steps,
            initial_lambda=self.initial_lambda,
            final_lambda=self.final_lambda,
            target_sparsity=self.target_sparsity,
            # epsilon=self.epsilon,
            learning_rate=self.learning_rate, 
            verbose=verbose
        )
        return self
        
    def transform(self, X, column_keep_indices=None, device=None):
        check_is_fitted(self)
        X = self._validate_data(X, accept_sparse=False, reset=False)
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
            for idx, batch in enumerate(tqdm(dataloader)):
                _, X_ = self.model_.forward(batch[0].to(device))
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