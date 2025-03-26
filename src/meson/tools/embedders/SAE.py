import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

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
              initial_lambda=1e-5,  # Start very small
              final_lambda=5.0,
              epsilon=0.05,
              target_sparsity=0.01):  # Target 1% activation
    """
    Train the SAE with modified sparsity control
    """
    # Scale dataset
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
        lr=5e-5,
        betas=(0.9, 0.999),
        weight_decay=0
    )
    
    losses = []
    sparsities = []
    recon_losses = []
    sparsity_losses = []
    
    step = 0
    pbar = tqdm(total=num_steps)
    
    # Adaptive lambda control
    current_lambda = initial_lambda
    
    for epoch in range((num_steps + len(dataloader) - 1) // len(dataloader)):
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
            normalized_weights = F.normalize(model.decoder.weight, p=2, dim=0)
            # Calculate cosine similarity matrix
            cosine_sim = torch.mm(normalized_weights.t(), normalized_weights)
            # Zero out diagonal (self-similarity)
            cosine_sim.fill_diagonal_(0)
            # Get maximum similarity
            max_cosine_sim = torch.max(cosine_sim)
            cosine_penalty = epsilon * max_cosine_sim

            loss = recon_loss + sparsity_loss + cosine_penalty
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Update learning rate
            if step > 0.8 * num_steps:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 5e-5 * (1 - (step - 0.8 * num_steps) / (0.2 * num_steps))
            
            # Adaptive lambda adjustment
            current_sparsity = (h > 0).float().mean().item()
            if step % 10 == 0:  # Adjust every 10 steps
                if current_sparsity < target_sparsity * 0.8:  # Too sparse
                    current_lambda = max(initial_lambda, current_lambda * 0.95)
                elif current_sparsity > target_sparsity * 1.2:  # Not sparse enough
                    current_lambda = min(final_lambda, current_lambda * 1.05)
            
            # Track metrics
            losses.append(loss.item())
            sparsities.append(current_sparsity)
            recon_losses.append(recon_loss.item())
            sparsity_losses.append(sparsity_loss.item())
            
            if step % 100 == 0:
                print(f"\nStep {step}")
                print(f"Total Loss: {losses[-1]:.4f}")
                print(f"Recon Loss: {recon_losses[-1]:.4f}")
                print(f"Sparsity Loss: {sparsity_losses[-1]:.4f}")
                print(f"Cosine Penalty: {cosine_penalty.item():.4f}")  # Added logging
                print(f"Sparsity (L0): {sparsities[-1]:.4f}")
                print(f"Learning rate: {optimizer.param_groups[0]['lr']:.2e}")
                print(f"Lambda: {current_lambda:.2e}")
                
            step += 1
            pbar.update(1)
            
    pbar.close()
    return {
        'losses': losses,
        'sparsities': sparsities,
        'recon_losses': recon_losses,
        'sparsity_losses': sparsity_losses
    }