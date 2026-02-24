import lazy_loader as lazy
__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
    'test': ['TestEmbedder'],
    'UNI': ['UNIEmbedder'],
    'UNI2': ['UNI2Embedder'],
    'Virchow2': ['Virchow2Embedder'],
    'SAE': ['SparseAutoencoder'],
    'KMeans': ['FrequencyRankedKMeans']
    }
)