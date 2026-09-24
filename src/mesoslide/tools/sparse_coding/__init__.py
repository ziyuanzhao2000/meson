"""Sparse coding of patch embeddings.

Sparse-coding models (`SparseAutoencoder`, `LocalityConstrainedCoding`) share
one fitted interface, which `feature_scorer`, `feature_extraction(sparse=True)`
and `fit_token_clusterer(scorer=model)` rely on:

    transform(X, column_keep_indices=None, device=None, *, progress_bar=...) -> (n, M) matrix

Column `i` of the output is written into `table.X` as `f"{prefix}_{i}"`.
"""

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
        '_feature_scorer': ['feature_scorer', 'feature_column_index'],
    }
)
