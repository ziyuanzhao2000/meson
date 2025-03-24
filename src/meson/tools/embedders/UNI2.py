from .model_manager import ModelManager
import os
import torch
from torch import nn
import timm
from huggingface_hub import login, hf_hub_download
from torchvision.transforms import v2

class UNI2ModelManager(ModelManager):
    @classmethod
    def _load_model_and_transform(cls):
        login(token=cls.get_token()) 
        print(f"Loading UNI2 model and transform, parameters are from {cls.download_dir}")
        local_dir = os.path.join(cls.download_dir, 'ckpts/uni2-h')
        os.makedirs(cls.download_dir, exist_ok=True) 
        hf_hub_download("MahmoodLab/UNI2-h", filename="pytorch_model.bin", local_dir=local_dir, force_download=True)
        timm_kwargs = {
            'model_name': 'vit_giant_patch14_224',
            'img_size': 224, 
            'patch_size': 14, 
            'depth': 24,
            'num_heads': 24,
            'init_values': 1e-5, 
            'embed_dim': 1536,
            'mlp_ratio': 2.66667*2,
            'num_classes': 0, 
            'no_embed_class': True,
            'mlp_layer': timm.layers.SwiGLUPacked, 
            'act_layer': torch.nn.SiLU, 
            'reg_tokens': 8, 
            'dynamic_img_size': True
            }
        model = timm.create_model(**timm_kwargs)
        model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"), strict=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        transform = v2.Compose([
                            v2.ToImage(),
                            v2.Resize(224),
                            v2.CenterCrop(224), # copied this from their Git repo but maybe it's redundant?
                            v2.ToDtype(torch.float32, scale=True),
                            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ])
        cls.model = model
        cls.transform = transform

class UNI2Embedder():
    def __init__(self):
        self.name = 'UNI2'
        self.embed_dim = 1536
        self.manager = UNI2ModelManager.initialize()
        self.model = self.manager.model
        self.transform = self.manager.transform

    def __call__(self, batch):
        """(B, C, Y, X) -> (B, embed_dim)"""
        batch = self.transform(batch)
        return self.model(batch)