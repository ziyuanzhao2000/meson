from torch import nn

class TestEmbedder(nn.Module):
    def __init__(self, embed_dim=1024):
        # lazy import
        from torchvision.transforms import v2
        import torch
        
        super().__init__()
        self.name = 'test'
        self.embed_dim = embed_dim
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.Resize(224),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, batch):
        """(B, C, Y, X) -> (B, embed_dim)"""
        batch = batch[:, :3, :, :]
        batch = self.transform(batch)
        return torch.rand(batch.shape[0], self.embed_dim)