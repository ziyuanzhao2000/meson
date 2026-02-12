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

        # Device setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        cls.model = model
        cls.transform = transform


class Virchow2Embedder():
    def __init__(self, token=None):
        self.name = 'Virchow2'
        self.embed_dim = 2560  # 1280 (class) + 1280 (mean patches)
        if token is not None:
            Virchow2ModelManager.set_token(token)
        self.manager = Virchow2ModelManager.initialize()
        self.model = self.manager.model
        self.transform = self.manager.transform

    def __call__(self, batch):
        """(B, C, Y, X) -> (B, embed_dim)"""
        batch = self.transform(batch)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            output = self.model(batch)
            
        # Process output per Virchow2 specs [2]
        class_token = output[:, 0]  # Class token
        patch_tokens = output[:, 5:]  # Skip register tokens
        return torch.cat([
            class_token, 
            patch_tokens.mean(dim=1)  # Average pooled patches
        ], dim=-1)
