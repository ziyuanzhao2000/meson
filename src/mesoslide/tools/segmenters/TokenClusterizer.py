import numpy as np
import torch
import cv2
from typing import Optional, Union
from tqdm import tqdm
import anndata
from sklearn.cluster import KMeans
class TokenClusterizer:
    """
    Token-level clustering and rasterization for vision transformer embeddings.
    
    Takes token embeddings from a ViT model, applies KMeans clustering,
    and upsamples cluster assignments to patch image resolution.
    
    Parameters
    ----------
    embedder : UNIEmbedder or similar
        Model with get_token_embeddings() method
    kmeans : FrequencyRankedKMeans or sklearn KMeans
        Fitted clustering model
    token_grid_size : int, default=14
        Spatial grid size of tokens (14×14 = 196 tokens for ViT-L/16)
    interpolation : str, default='nearest'
        Interpolation method for upsampling ('nearest' or 'bilinear')
    cluster_order : np.ndarray, optional
        Custom ordering for cluster IDs. If provided, remaps cluster labels
        according to this order before rasterization.
    feature_name: xxx
    device : str, default='cuda'
        Device for computation
    
    Examples
    --------
    >>> from meson.tools.embedders import UNIEmbedder
    >>> from meson.tools.segmenters import TokenClusterizer
    >>> import torch
    >>> import joblib
    >>> 
    >>> # Load model and kmeans
    >>> uni = UNIEmbedder(token='your_hf_token')
    >>> kmeans = joblib.load('feature_classifier.joblib')
    >>> 
    >>> # Create clusterizer
    >>> clusterizer = TokenClusterizer(
    >>>     embedder=uni,
    >>>     kmeans=kmeans,
    >>>     token_grid_size=14
    >>> )
    >>> 
    >>> # Process batch
    >>> images = torch.randn(8, 3, 224, 224)  # batch of 8 images
    >>> cluster_masks = clusterizer(images, output_size=(224, 224))
    >>> # Returns: (8, 224, 224) uint8 array with cluster IDs
    """
    
    def __init__(
        self,
        embedder,
        kmeans,
        token_grid_size: int = 14,
        interpolation: str = 'nearest',
        cluster_order: Optional[np.ndarray] = None,
        feature_name: str = '',
        device: str = 'cuda'
    ):
        self.embedder = embedder
        self.kmeans = kmeans
        self.token_grid_size = token_grid_size
        self.interpolation = interpolation
        self.cluster_order = cluster_order
        self.feature_name = feature_name
        self.device = device
        
        # Validate interpolation
        if interpolation not in ['nearest', 'bilinear']:
            raise ValueError(f"interpolation must be 'nearest' or 'bilinear', got {interpolation}")
        
        self.cv2_interp = (
            cv2.INTER_NEAREST_EXACT if interpolation == 'nearest' 
            else cv2.INTER_LINEAR
        )
    
    def _cluster_tokens(self, token_embeddings: torch.Tensor) -> np.ndarray:
        """
        Apply KMeans clustering to token embeddings.
        
        Parameters
        ----------
        token_embeddings : torch.Tensor
            Shape (B, N_tokens, embed_dim)
        
        Returns
        -------
        cluster_maps : np.ndarray
            Shape (B, grid_h, grid_w) with cluster IDs
        """
        B, N, D = token_embeddings.shape
        assert N == self.token_grid_size ** 2, \
            f"Expected {self.token_grid_size**2} tokens, got {N}"
        
        # Flatten and predict
        token_embeddings_np = token_embeddings.cpu().numpy().astype(np.float64)

        cluster_maps = []
        
        for i in range(B):
            flat_tokens = token_embeddings_np[i]  # (N, D)
            labels = self.kmeans.predict(flat_tokens)
            
            # Reshape to grid
            cluster_map = labels.reshape(self.token_grid_size, self.token_grid_size)
            cluster_maps.append(cluster_map)
        
        return np.array(cluster_maps, dtype=np.uint8)
    
    def _rasterize(self, cluster_maps: np.ndarray, output_size: tuple) -> np.ndarray:
        """
        Upsample cluster maps to target resolution.
        
        Parameters
        ----------
        cluster_maps : np.ndarray
            Shape (B, grid_h, grid_w)
        output_size : tuple
            Target (height, width)
        
        Returns
        -------
        rasterized : np.ndarray
            Shape (B, H, W) with upsampled cluster IDs
        """
        B = cluster_maps.shape[0]
        H, W = output_size
        
        rasterized = np.zeros((B, H, W), dtype=np.uint8)
        
        for i in range(B):
            cluster_map = cluster_maps[i]
            
            # Remap if cluster_order provided
            if self.cluster_order is not None:
                cluster_map = self.cluster_order[cluster_map]
            
            # Upsample
            upsampled = cv2.resize(
                cluster_map,
                (W, H),
                interpolation=self.cv2_interp
            )
            rasterized[i] = upsampled
        
        return rasterized
    
    def __call__(
        self,
        images: Union[torch.Tensor, np.ndarray, 'anndata._core.anndata.AnnData'],
        sdata: Optional['spatialdata.SpatialData'] = None,
        output_size: Optional[tuple] = None,
        batch_size: int = 16,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate cluster masks for a batch of images.
        
        Parameters
        ----------
        images : torch.Tensor, np.ndarray, or pd.DataFrame
            Input images. Can be:
            - torch.Tensor: (N, C, H, W) in [0, 1]
            - np.ndarray: (N, H, W, C) in [0, 255]
            - pd.DataFrame: patch table with columns [image, xmin, xmax, ymin, ymax]
        sdata : spatialdata.SpatialData, optional
            Required if images is a DataFrame. Used to extract patches.
        output_size : tuple, optional
            Target (height, width) for masks. If None, uses input image size.
        batch_size : int, default=16
            Batch size for processing
        show_progress : bool, default=True
            Whether to show progress bar
        
        Returns
        -------
        cluster_masks : np.ndarray
            Shape (N, H, W), dtype uint8
            Each value is a cluster ID
        
        Examples
        --------
        >>> # Method 1: Direct numpy array
        >>> patches = np.random.randint(0, 255, (10, 224, 224, 3), dtype=np.uint8)
        >>> masks = clusterizer(patches)
        
        >>> # Method 2: From patch table
        >>> top_patches_df = meson.select_top_patches(sdata, ...)
        >>> masks = clusterizer(top_patches_df, sdata=sdata)
        """
        # Handle DataFrame input - extract patches
        if hasattr(images, 'obs'):  # It's a AnnData object
            if sdata is None:
                raise ValueError("sdata must be provided when images is a DataFrame")
            
            # Import here to avoid circular dependency
            from meson.preprocessing import extract_patches
            
            if show_progress:
                print(f"Extracting {len(images)} patches...")
            images = extract_patches(sdata, images)
            
        # Convert to tensor if needed
        if isinstance(images, np.ndarray):
            # Assume (N, C, H, W) uint8 or uint16 format
            images = torch.from_numpy(images).float() / 255.0

        if output_size is None:
            output_size = (images.shape[2], images.shape[3])
        
        # Process in batches
        all_cluster_maps = []
        n_batches = int(np.ceil(len(images) / batch_size))
        
        iterator = range(n_batches)
        if show_progress:
            iterator = tqdm(iterator, desc="Clustering patches")
        
        self.embedder.model.eval()
        with torch.inference_mode():
            for i in iterator:
                batch = images[i * batch_size:(i + 1) * batch_size].to(self.device)
                
                # Get token embeddings
                token_embeds = self.embedder.get_token_embeddings(batch, remove_cls=True)
                
                # Cluster
                cluster_maps = self._cluster_tokens(token_embeds)
                all_cluster_maps.append(cluster_maps)
        
        # Concatenate all batches
        all_cluster_maps = np.concatenate(all_cluster_maps, axis=0)
        
        # Rasterize
        rasterized_masks = self._rasterize(all_cluster_maps, output_size)
        
        return rasterized_masks
    
    def fit(
        self,
        sdata: 'spatialdata.SpatialData',
        patch_table_names: Union[str, list],
        feature_name: str,
        n_positive: int = 100,
        n_negative: int = 100,
        batch_size: int = 16,
        show_progress: bool = True,
        take_every: Union[int, None] = None,
    ) -> np.ndarray:
        """
        Compute cluster order based on differential abundance between positive and negative patches.
        
        This method:
        1. Selects positive (high-scoring) and negative (zero-score) patches for a feature
        2. Extracts image patches and computes token embeddings
        3. Predicts cluster labels for all tokens
        4. Computes differential cluster frequencies (positive - negative)
        5. Ranks clusters by differential abundance
        6. Updates self.cluster_order and returns the ordering
        
        This is useful for identifying which tissue structures (clusters) are
        most enriched in patches where a specific SAE feature is active.
        
        Parameters
        ----------
        sdata : SpatialData
            Spatial data object containing images and patch tables
        patch_table_names : str or list
            Name(s) of patch tables to sample from
        feature_name : str
            Feature name to use for patch selection (e.g., 'UNI_SAE_12345')
        n_positive : int, default=100
            Number of positive patches to sample
        n_negative : int, default=100
            Number of negative patches to sample
        batch_size : int, default=16
            Batch size for processing
        show_progress : bool, default=True
            Whether to show progress bars
        
        Returns
        -------
        cluster_order : np.ndarray
            Indices that sort clusters by differential abundance (high to low).
            Also stored in self.cluster_order.
        
        Examples
        --------
        >>> # Compute cluster order for a specific SAE feature
        >>> clusterizer.compute_and_set_cluster_order_from_feature(
        ...     sdata=sdata,
        ...     patch_table_names='all_patches',
        ...     feature_name='UNI_SAE_12345',
        ...     n_positive=100,
        ...     n_negative=100
        ... )
        >>> # Now the clusterizer will use this ordering when rasterizing
        >>> masks = clusterizer(images)
        """
        from meson._patch_selector import select_top_patches, select_negative_patches
        from meson.preprocessing import extract_patches
        
        if show_progress:
            print(f"Selecting patches for feature '{feature_name}'...")
        
        # Get positive patches (evenly sampled from high-scoring patches)
        positive_patches_anndata = select_top_patches(
            sdata,
            patch_table_names=patch_table_names,
            feature_name=feature_name,
            n=n_positive,
            min_score=0,  # Only positive scores
            take_every=take_every  # Auto-compute stride
        )
        
        # Get negative patches (evenly sampled from zero-score patches)
        negative_patches_anndata = select_negative_patches(
            sdata,
            patch_table_names=patch_table_names,
            feature_name=feature_name,
            n=n_negative,
            take_every=None  # Auto-compute stride
        )
        
        if show_progress:
            print(f"Extracting {len(positive_patches_anndata)} positive and {len(negative_patches_anndata)} negative patches...")
        
        # Extract actual image patches
        positive_patches = extract_patches(sdata, positive_patches_anndata, 
                                          channel_first=True, progress_bar=show_progress)
        negative_patches = extract_patches(sdata, negative_patches_anndata,
                                          channel_first=True, progress_bar=show_progress)
        
        # Convert to tensors
        positive_patches = torch.from_numpy(positive_patches).float() / 255.0
        negative_patches = torch.from_numpy(negative_patches).float() / 255.0
        
        if show_progress:
            print("Computing token embeddings...")
        
        # Extract token embeddings for both sets
        all_patches = torch.cat([positive_patches, negative_patches], dim=0)
        all_tokens = []
        
        n_batches = int(np.ceil(len(all_patches) / batch_size))
        iterator = range(n_batches)
        if show_progress:
            iterator = tqdm(iterator, desc="Extracting tokens")
        
        self.embedder.model.eval()
        with torch.inference_mode():
            for i in iterator:
                batch = all_patches[i * batch_size:(i + 1) * batch_size].to(self.device)
                token_embeds = self.embedder.get_token_embeddings(batch, remove_cls=True)
                
                # Flatten: (B, N, D) -> (B*N, D)
                flat = token_embeds.reshape(-1, token_embeds.shape[-1]).cpu().numpy()
                all_tokens.append(flat)
        
        all_tokens = np.concatenate(all_tokens, axis=0).astype(np.float64)
        
        # Split back into positive and negative tokens
        n_pos = len(positive_patches)
        n_neg = len(negative_patches)
        tokens_per_patch = self.token_grid_size ** 2
        pos_token_count = n_pos * tokens_per_patch
        neg_token_count = n_neg * tokens_per_patch
        
        positive_tokens = all_tokens[:pos_token_count]
        negative_tokens = all_tokens[pos_token_count:pos_token_count + neg_token_count]
        
        if show_progress:
            print("Predicting cluster labels...")
        
        # Fit KMeans if not already fitted
        if not hasattr(self.kmeans, 'cluster_centers_'):
            if show_progress:
                print("Fitting KMeans on all tokens...")
            self.kmeans = KMeans(n_clusters=3, random_state=0).fit(positive_tokens)
        
        # Predict cluster labels
        positive_labels = self.kmeans.predict(positive_tokens)
        negative_labels = self.kmeans.predict(negative_tokens)
        
        # Get number of clusters
        n_clusters = len(np.unique(np.concatenate([positive_labels, negative_labels])))
        if hasattr(self.kmeans, 'n_clusters'):
            n_clusters = self.kmeans.n_clusters
        
        # Compute normalized frequencies
        positive_counts = np.bincount(positive_labels, minlength=n_clusters)
        negative_counts = np.bincount(negative_labels, minlength=n_clusters)
        
        percentage_positive = positive_counts / (positive_counts.sum() + 1e-12)
        percentage_negative = negative_counts / (negative_counts.sum() + 1e-12)
        
        # Compute differential abundance
        diff_percentage = percentage_positive - percentage_negative
        
        # Rank clusters by differential abundance (high to low)
        cluster_order = np.argsort(np.argsort(-diff_percentage))
        
        # Store and return
        self.cluster_order = cluster_order
        
        if show_progress:
            print(f"Cluster order computed and stored. Top 3 enriched clusters: {cluster_order[:3]}")
            print(f"Differential abundances: {diff_percentage[cluster_order[:3]]}")
        
        return self