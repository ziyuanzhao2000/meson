from .test import TestEmbedder
from .UNI import UNIEmbedder
from .UNI2 import UNI2Embedder
from .SAE import SparseAutoencoder

__all__ = [TestEmbedder, UNIEmbedder, UNI2Embedder, SparseAutoencoder]

available_embedders = {
    "test": TestEmbedder,
    "UNI": UNIEmbedder,
    "UNI2": UNI2Embedder,
    "SAE": SparseAutoencoder
}