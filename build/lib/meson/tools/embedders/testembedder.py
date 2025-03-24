from torch import nn
from torchvision.transforms import v2
import torch

class TestEmbedder(nn.Module):
    def __init__(self, embed_dim=1024):
        super().__init__()
        self.name = 'test'
        self.embed_dim = embed_dim
        self.prepr

    def __call__(self, x):
        """(B, C, Y, X) -> (B, embed_dim)"""
        return torch.rand(x.shape[0], self.embed_dim)