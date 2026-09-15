import lazy_loader as lazy
__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        '_feature_selector': ['FeatureSelector'],
        '_feature_clusterer': ['FeatureClusterer'],
        '_sae': ['SimpleAutoencoder', 'train_simple_sae', 'SparseAutoencoder'],
        '_llc': ['LLCModel', 'fit_codebook', 'LocalityConstrainedCoding'],
        '_kmeans_backends': ['TorchMiniBatchKMeans', 'fit_kmeans_backend'],
        '_feature_aggregator': ['FeatureGroupAggregator', 'aggregate_feature_groups'],
    }
)
