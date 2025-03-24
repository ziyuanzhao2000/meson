from .test import TestEmbedder
from .UNI import UNIEmbedder
from .UNI2 import UNI2Embedder

__all__ = [TestEmbedder, UNIEmbedder, UNI2Embedder]

available_embedders = {
    "test": TestEmbedder,
    "UNI": UNIEmbedder,
    "UNI2": UNI2Embedder
}