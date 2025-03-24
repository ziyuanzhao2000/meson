from .test import TestEmbedder
from .UNI import UNIEmbedder

__all__ = [TestEmbedder, UNIEmbedder]

available_embedders = {
    "test": TestEmbedder,
    "UNI": UNIEmbedder
}