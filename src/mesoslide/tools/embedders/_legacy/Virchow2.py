from .model_manager import ModelManager
import timm
import torch
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked
from huggingface_hub import login
import os


class Virchow2ModelManager(ModelManager):
    @classmethod
    def _load_model_and_transform(cls):
        login(token=cls.get_token())
        print(f"Loading Virchow2 model and transform from {cls.download_dir}")
        
        # Model configuration from Hugging Face card
        model = timm.create_model(
            "hf-hub:paige-ai/Virchow2",
            pretrained=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU
        )
        
        # Setup transforms using timm's config system
        data_config = resolve_data_config(model.pretrained_cfg, model=model)
        transform = create_transform(**data_config)
        
        model = model.to(cls.device)
        model.eval()

        cls.model = model
        cls.transform = transform


class Virchow2Embedder():
    def __init__(self, token=None, device=None):
        self.name = 'Virchow2'
        self.embed_dim = 2560  # 1280 (class) + 1280 (mean patches)
        if token is not None:
            Virchow2ModelManager.set_token(token)
        self.manager = Virchow2ModelManager.initialize()
        self.manager.set_device(device)
        
    @property
    def model(self):
        return Virchow2ModelManager.model

    @property
    def transform(self):
        return Virchow2ModelManager.transform
    
    @property
    def device(self):
        return Virchow2ModelManager._resolve_device()

    def to(self, device):
        """Switch device. Returns self for chaining."""
        Virchow2ModelManager.set_device(device)
        return self
    
    def __call__(self, batch):
        """(B, C, Y, X) -> (B, embed_dim)"""
        batch = self.transform(batch).to(self.device)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            output = self.model(batch)
            
        # Process output per Virchow2 specs [2]
        class_token = output[:, 0]  # Class token
        patch_tokens = output[:, 5:]  # Skip register tokens
        return torch.cat([
            class_token, 
            patch_tokens.mean(dim=1)  # Average pooled patches
        ], dim=-1)
