# /n/scratch/users/z/ziz531/meson/src/meson/tools/embedders/FrequencyRankedKMeans.py

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, ClusterMixin
from sklearn.cluster import KMeans
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_random_state
from typing import Optional


class FrequencyRankedKMeans(TransformerMixin, BaseEstimator, ClusterMixin):
    """
    K-Means clustering with frequency-based label remapping.
    
    Wraps scikit-learn KMeans and remaps cluster labels by frequency:
    the most common cluster becomes label 0, second most common becomes 1, etc.
    
    Parameters
    ----------
    n_clusters : int, default=25
        The number of clusters to form and centroids to generate.
    normalize : bool, default=True
        Whether to L2-normalize embeddings before clustering.
    batch_size : int, default=2048
        Batch size for transform operations (for consistency with SAE interface).
    random_state : int, RandomState instance or None, default=None
        Random state for reproducibility.
    **kmeans_kwargs : dict
        Additional keyword arguments passed to sklearn.cluster.KMeans.
    
    Attributes
    ----------
    kmeans_ : KMeans
        The fitted KMeans model.
    label_mapping_ : dict
        Mapping from original cluster IDs to frequency-ranked IDs.
    sorted_cluster_ids_ : np.ndarray
        Original cluster IDs sorted by frequency (descending).
    sorted_counts_ : np.ndarray
        Cluster counts sorted by frequency (descending).
    embed_dim_ : int
        Dimensionality of input embeddings (set during fit).
    
    Examples
    --------
    >>> from meson.tools.embedders import FrequencyRankedKMeans
    >>> import numpy as np
    >>> 
    >>> # Generate sample data
    >>> X = np.random.randn(1000, 128)
    >>> 
    >>> # Fit and transform
    >>> model = FrequencyRankedKMeans(n_clusters=25, random_state=42)
    >>> model.fit(X)
    >>> labels = model.transform(X)
    >>> 
    >>> # Most common cluster is now labeled 0
    >>> print(f"Cluster 0 has {np.sum(labels == 0)} samples")
    """
    
    def __init__(
        self,
        n_clusters: int = 25,
        normalize: bool = True,
        batch_size: int = 2048,
        random_state: Optional[int] = None,
        **kmeans_kwargs
    ):
        self.n_clusters = n_clusters
        self.normalize = normalize
        self.batch_size = batch_size
        self.random_state = random_state
        self.kmeans_kwargs = kmeans_kwargs
    
    def _normalize_embeddings(self, X):
        """L2-normalize embeddings to unit length."""
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        return X / norms
    
    def fit(self, X, y=None):
        """
        Fit K-Means model and compute frequency-based label mapping.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training embeddings.
        y : Ignored
            Not used, present for API consistency.
        
        Returns
        -------
        self : FrequencyRankedKMeans
            Fitted estimator.
        """
        self.random_state_ = check_random_state(self.random_state)
        X = self._validate_data(X, accept_sparse=False)
        
        assert len(X.shape) == 2, f"Expected 2D array, got shape {X.shape}"
        self.embed_dim_ = X.shape[1]
        
        # Normalize if requested
        if self.normalize:
            X_processed = self._normalize_embeddings(X)
        else:
            X_processed = X
        
        # Fit KMeans
        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state_,
            **self.kmeans_kwargs
        )
        original_labels = self.kmeans_.fit_predict(X_processed)
        
        # Compute cluster frequencies
        cluster_ids, counts = np.unique(original_labels, return_counts=True)
        
        # Sort by frequency (descending)
        sort_indices = np.argsort(-counts)
        self.sorted_cluster_ids_ = cluster_ids[sort_indices]
        self.sorted_counts_ = counts[sort_indices]
        
        # Create mapping: original_id -> frequency_rank_id
        self.label_mapping_ = {
            old_id: new_id 
            for new_id, old_id in enumerate(self.sorted_cluster_ids_)
        }
        
        # Print cluster statistics
        print(f"Fitted FrequencyRankedKMeans with {self.n_clusters} clusters")
        print(f"Cluster counts (ranked by frequency):")
        for i, (cluster_id, count) in enumerate(
            zip(self.sorted_cluster_ids_, self.sorted_counts_)
        ):
            pct = count / len(X) * 100
            print(f"  Rank {i} (original cluster {cluster_id}): "
                  f"{count:,} samples ({pct:.1f}%)")
        
        return self
    
    def transform(self, X):
        """
        Predict cluster labels with frequency-based remapping.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Embeddings to cluster.
        
        Returns
        -------
        labels : np.ndarray of shape (n_samples,)
            Frequency-ranked cluster labels (0 = most common, etc.)
        """
        check_is_fitted(self)
        X = self._validate_data(X, accept_sparse=False, reset=False)
        
        # Normalize if requested
        if self.normalize:
            X_processed = self._normalize_embeddings(X)
        else:
            X_processed = X
        
        # Predict original labels
        original_labels = self.kmeans_.predict(X_processed)
        
        # Remap to frequency-ranked labels
        remapped_labels = np.array([
            self.label_mapping_[label] 
            for label in original_labels
        ])
        
        return remapped_labels
    
    def fit_transform(self, X, y=None):
        """
        Fit model and return frequency-ranked labels.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training embeddings.
        y : Ignored
            Not used, present for API consistency.
        
        Returns
        -------
        labels : np.ndarray of shape (n_samples,)
            Frequency-ranked cluster labels.
        """
        return self.fit(X, y).transform(X)
    
    def predict(self, X):
        """Alias for transform (for compatibility with clustering API)."""
        return self.transform(X)