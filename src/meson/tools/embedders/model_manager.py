
import os
from typing import Optional
from getpass import getpass

class ModelManager:
    _instance = None
    model = None
    transform = None
    download_dir = None
    _token = None

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