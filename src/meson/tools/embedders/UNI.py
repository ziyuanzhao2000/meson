from .model_manager import ModelManager
import timm
import torch
from huggingface_hub import login, hf_hub_download
from torchvision.transforms import v2
import os

class UNIModelManager(ModelManager):
    @classmethod
    def _load_model_and_transform(cls):

        login(token=cls.get_token()) 
        print(f"Loading UNI model and transform, parameters are from {cls.download_dir}")
        local_dir = os.path.join(cls.download_dir, 'ckpts/vit_large_patch16_224.dinov2.uni_mass100k')
        os.makedirs(cls.download_dir, exist_ok=True) 
        hf_hub_download("MahmoodLab/UNI", filename="pytorch_model.bin", local_dir=local_dir)

        model = timm.create_model(
            "vit_large_patch16_224", 
            img_size=224, 
            patch_size=16, 
            init_values=1e-5, 
            num_classes=0, 
            dynamic_img_size=True
        )
        model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"), strict=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        transform = v2.Compose([
                            v2.ToImage(),
                            v2.Resize(224),
                            v2.ToDtype(torch.float32, scale=True),
                            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ])

        cls.model = model
        cls.transform = transform
        cls.loaded = True

class UNIEmbedder():
    def __init__(self, token=None):
        self.name = 'UNI'
        self.embed_dim = 1024
        if token is not None:
            UNIModelManager.set_token(token)
        self.manager = UNIModelManager.initialize()
        self.model = self.manager.model
        self.transform = self.manager.transform

    def __call__(self, batch):
        """(B, C, Y, X) -> (B, embed_dim)"""
        batch = self.transform(batch)
        return self.model(batch)