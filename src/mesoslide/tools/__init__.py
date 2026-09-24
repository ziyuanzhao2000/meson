import lazy_loader as lazy
__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
    'segmenters': ['GenericSegmenter', 'TokenClusterer', 'fit_token_clusterer', 'predict_token_labels',
                   'TokenClusterizer', 'adaptive_sample_wsi'],
    'sparse_coding': ['FeatureSelector', 'FeatureClusterer', 'SparseAutoencoder', 'LocalityConstrainedCoding',
                       'FeatureGroupAggregator', 'aggregate_feature_groups', 'feature_scorer'],
    '_feature_extraction': ['feature_extraction', 'embed_patch'],
    '_token_ablation': ['token_ablation', 'summarize_token_ablation'],
    '_cell_import': ['add_cell_polygons', 'add_cell_phenotypes', 'add_cells'],
    }
)
