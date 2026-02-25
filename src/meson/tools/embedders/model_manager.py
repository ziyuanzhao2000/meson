
import os
from typing import Optional
from getpass import getpass
import torch 

class ModelManager:
    _instance = None
    model = None
    transform = None
    download_dir = None
    _token = None
    device = None 

    @classmethod
    def set_token(cls, token):
        cls._token = token

    @classmethod
    def get_token(cls):
        if cls._token is None:
            # Check for environment variable
            cls._token = os.environ.get('HF_TOKEN')
            
            # If not found, prompt the user
            if cls._token is None:
                cls._token = getpass("Enter your Hugging Face token: ")
        
        return cls._token
    
    @classmethod
    def set_device(cls, device=None):
        """Move model to a new device and update state."""
        if device is None:
            device = cls._resolve_device()
        cls.device = torch.device(device)
        if cls.model is not None:
            cls.model = cls.model.to(cls.device)
            print(f"{cls.__name__}: model moved to {cls.device}")

    @classmethod
    def _resolve_device(cls, device=None):
        """Resolve device: explicit arg > class attr > auto-detect."""
        if device is not None:
            return torch.device(device)
        if cls.device is not None:
            return cls.device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @classmethod
    def initialize(cls, download_dir: Optional[str] = None):
        # try:
            if cls._instance is None or not cls.loaded:
                cls.loaded = False
                cls._instance = cls()
                cls.download_dir = download_dir or os.getcwd()
                cls._load_model_and_transform()
            return cls._instance
        # except:
        #     print("Failed to initialize the model manager! Please try again")

    @classmethod
    def _load_model_and_transform(cls):
        print(f"Loading model and transform, parameters are from {cls.download_dir}")
        cls.model = "Pretrained Model"
        cls.transform = "Input Transform"
        cls.loaded = True