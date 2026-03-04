import lazy_loader as lazy
__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
    'embedders': ['TestEmbedder', 'UNIEmbedder', 'UNI2Embedder', 'SparseAutoencoder'],
    'segmenters': ['GenericSegmenter', 'TokenClusterizer', 'adaptive_sample_wsi'],
    '_embed_patch': ['embed_patch'],
    '_sae_feature_selector': ['SAEFeatureSelector']
    }
)
