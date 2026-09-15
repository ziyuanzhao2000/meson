import lazy_loader as lazy
__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
    'segmenters': ['GenericSegmenter', 'TokenClusterizer', 'adaptive_sample_wsi'],
    'sparse_coding': ['FeatureSelector', 'FeatureClusterer', 'SparseAutoencoder', 'LocalityConstrainedCoding',
                       'FeatureGroupAggregator', 'aggregate_feature_groups'],
    '_feature_extraction': ['feature_extraction', 'embed_patch'],
    }
)
