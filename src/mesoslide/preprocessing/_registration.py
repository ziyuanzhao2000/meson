"""
Image registration utilities for transforming coordinates between modalities.
"""

import math
import joblib

import numpy as np
from scipy.ndimage import map_coordinates
import anndata as ad
from meson._tifffile_wsi import TiffFile


class ForwardTransform:
    """
    Forward transform for mapping coordinates from source to target image space.
    
    This class encapsulates affine + deformable registration transforms and handles
    all the coordinate scaling and padding logic needed to transform points from
    the source coordinate system (e.g., H&E) to the target coordinate system (e.g., CyCIF).
    
    Parameters
    ----------
    affine_tform : skimage.transform.AffineTransform
        Affine transformation object.
    deform_dfield : ndarray of shape (H, W, 2)
        Deformation field with (dx, dy) displacements at each pixel.
    source_shape : tuple of (height, width)
        Shape of the source image.
    target_shape : tuple of (height, width)
        Shape of the target image.
    pad_source : list of [(top, bottom), (left, right)]
        Padding applied to source image.
    pad_target : list of [(top, bottom), (left, right)]
        Padding applied to target image.
    """
    
    def __init__(self, affine_tform, deform_dfield, source_shape, target_shape, 
                 pad_source, pad_target):
        """Initialize the ForwardTransform with all transformation components."""
        self.affine_tform = affine_tform
        self.deform_field_x = deform_dfield[..., 0]
        self.deform_field_y = deform_dfield[..., 1]
        
        self.source_h, self.source_w = source_shape
        self.target_h, self.target_w = target_shape
        self.deform_h, self.deform_w = deform_dfield.shape[:2]
        
        self.pad_source = pad_source
        self.pad_target = pad_target
        
    @classmethod
    def from_file(cls, transform_path, source_image_path, target_image_path):
        """
        Load a forward transform from disk and set up coordinate transformation.
        
        Parameters
        ----------
        transform_path : str
            Path to the saved transform file (.pkl with affine_tform and deform_dfield).
        source_image_path : str
            Path to the source image (e.g., H&E) to get dimensions.
        target_image_path : str
            Path to the target image (e.g., CyCIF) to get dimensions.
            
        Returns
        -------
        ForwardTransform
            Initialized transform object ready for coordinate transformation.
            
        Examples
        --------
        >>> ft = ForwardTransform.from_file(
        ...     '/path/to/LSP24530.pkl',
        ...     '/path/to/LSP24530.ome.tif',
        ...     '/path/to/LSP24531_DNA1.ome.tif'
        ... )
        >>> transformed = ft.transform_points(points)
        """
        # Load the transform
        affine_tform, deform_dfield = joblib.load(transform_path)
        
        # Load images to get dimensions
        source_tiff = TiffFile(source_image_path)
        target_tiff = TiffFile(target_image_path)
        
        # Get source dimensions (H&E typically has 3 channels)
        source_shape_raw = source_tiff[0][0].shape
        if len(source_shape_raw) == 3:
            source_h, source_w, _ = source_shape_raw
        else:
            source_h, source_w = source_shape_raw
            
        # Get target dimensions
        target_shape_raw = target_tiff[0][0].shape
        if len(target_shape_raw) == 3:
            target_h, target_w, _ = target_shape_raw
        else:
            target_h, target_w = target_shape_raw
        
        source_tiff.close()
        target_tiff.close()
        
        # Calculate padding
        pad_source, pad_target = cls._calculate_padding(
            (source_h, source_w), 
            (target_h, target_w)
        )
        
        return cls(
            affine_tform=affine_tform,
            deform_dfield=deform_dfield,
            source_shape=(source_h, source_w),
            target_shape=(target_h, target_w),
            pad_source=pad_source,
            pad_target=pad_target
        )
    
    @staticmethod
    def _calculate_padding(source_shape, target_shape):
        """
        Calculate padding needed to match two image sizes.
        
        Parameters
        ----------
        source_shape : tuple of (height, width)
            Source image dimensions.
        target_shape : tuple of (height, width)
            Target image dimensions.
            
        Returns
        -------
        pad_source : list of [(top, bottom), (left, right)]
            Padding for source image.
        pad_target : list of [(top, bottom), (left, right)]
            Padding for target image.
        """
        source_h, source_w = source_shape
        target_h, target_w = target_shape
        
        pad_source = [(0, 0), (0, 0)]
        pad_target = [(0, 0), (0, 0)]
        
        # Y dimension
        if source_h > target_h:
            pad_size = source_h - target_h
            pad_target[0] = (math.floor(pad_size / 2), math.ceil(pad_size / 2))
        elif source_h < target_h:
            pad_size = target_h - source_h
            pad_source[0] = (math.floor(pad_size / 2), math.ceil(pad_size / 2))
        
        # X dimension
        if source_w > target_w:
            pad_size = source_w - target_w
            pad_target[1] = (math.floor(pad_size / 2), math.ceil(pad_size / 2))
        elif source_w < target_w:
            pad_size = target_w - source_w
            pad_source[1] = (math.floor(pad_size / 2), math.ceil(pad_size / 2))
        
        return pad_source, pad_target
    
    def transform_points(self, points):
        """
        Transform points from source coordinates to target coordinates.
        
        This applies the full registration pipeline:
        1. Add source padding offset
        2. Scale to deformation field resolution
        3. Apply deformation field
        4. Apply inverse affine transform
        5. Scale back to full resolution
        6. Remove target padding offset
        
        Parameters
        ----------
        points : ndarray of shape (N, 2) or (..., 2)
            Points in source coordinate system as (x, y).
            
        Returns
        -------
        transformed : ndarray of same shape as input
            Points in target coordinate system as (x, y).
            
        Examples
        --------
        >>> # Transform patch centroids
        >>> centers = np.array([[1000, 2000], [1500, 2500]])
        >>> centers_transformed = ft.transform_points(centers)
        
        >>> # Transform arrays of arbitrary shape
        >>> corners = np.random.rand(100, 4, 2) * 5000  # 100 patches, 4 corners each
        >>> corners_transformed = ft.transform_points(corners)
        """
        # Work with a copy to avoid modifying input
        points = points.copy().astype(np.float64)
        
        # Step 1: Apply source padding offset (x, y) order
        points -= np.array([self.pad_source[1][0], self.pad_source[0][0]])
        
        # Step 2: Scale to deformation field resolution
        scale_to_deform = np.array([
            self.deform_w / self.source_w, 
            self.deform_h / self.source_h
        ])
        points = points * scale_to_deform

        # Step 3: Sample deformation field
        # map_coordinates expects (y, x) indexing
        py, px = points[..., 1], points[..., 0]
        dx = map_coordinates(self.deform_field_x, [py, px], order=1, mode='nearest')
        dy = map_coordinates(self.deform_field_y, [py, px], order=1, mode='nearest')
        deformation = np.stack([dx, dy], axis=-1)
        
        # Step 4: Apply inverse affine transform
        shape = deformation.shape
        deformed_points = deformation + points
        transformed = self.affine_tform.inverse(
            deformed_points.reshape(-1, 2)
        ).reshape(shape)
        # Step 5: Scale back to full target resolution
        scale_from_deform = np.array([
            self.source_w / self.deform_w,
            self.source_h / self.deform_h
        ])
        transformed = transformed * scale_from_deform
        
        # Step 6: Remove target padding offset
        transformed -= np.array([self.pad_target[1][0], self.pad_target[0][0]])
        
        return transformed
    
    def transform_patch_table(
            self,
            patch_table: ad.AnnData,
            new_patch_size: int
        ) -> ad.AnnData:
        """
        Transform patch bounding boxes to target coordinate system.
        
        This function:
        1. Extracts patch centroids from the source patch table
        2. Transforms them using the forward transform
        3. Creates new square bounding boxes centered on transformed centroids
        
        Parameters
        ----------
        patch_table : AnnData
            Patch table with .obs containing xmin, xmax, ymin, ymax in source coordinates.
        forward_transform : ForwardTransform
            Transform object for mapping coordinates from source to target space.
        new_patch_size : int
            Size of square patches in target coordinate system.
            
        Returns
        -------
        transformed_table : AnnData
            New patch table with updated bounding boxes in target coordinates.
            
        Examples
        --------
        >>> # Get top patches for a feature
        >>> top_patches = select_top_patches(patches, 'UNI_SAE_12345', n=100)
        
        >>> # Load transform
        >>> ft = ForwardTransform.from_file('transform.pkl', 'hne.tif', 'cycif.tif')
        
        >>> # Transform to CyCIF coordinate system with 377px patches
        >>> cycif_patches = transform_patch_table(top_patches, ft, new_patch_size=377)
        """
        forward_transform = self  # For clarity
        
        # Extract centroids from source patches
        xmin = patch_table.obs['xmin'].to_numpy()
        xmax = patch_table.obs['xmax'].to_numpy()
        ymin = patch_table.obs['ymin'].to_numpy()
        ymax = patch_table.obs['ymax'].to_numpy()
        
        centroids = np.stack([
            (xmin + xmax) / 2,
            (ymin + ymax) / 2
        ], axis=1)
        
        # Transform centroids to target space
        transformed_centroids = forward_transform.transform_points(centroids)
        
        # Create new square bounding boxes
        half_size = new_patch_size // 2
        
        new_obs = patch_table.obs.copy()
        new_obs['xmin'] = transformed_centroids[:, 0] - half_size
        new_obs['xmax'] = transformed_centroids[:, 0] + half_size
        new_obs['ymin'] = transformed_centroids[:, 1] - half_size
        new_obs['ymax'] = transformed_centroids[:, 1] + half_size
        
        # Create new AnnData with transformed coordinates
        transformed_table = ad.AnnData(
            X=patch_table.X.copy(),
            obs=new_obs,
            var=patch_table.var.copy()
        )
        
        return transformed_table

    def __repr__(self):
        return (
            f"ForwardTransform(\n"
            f"  source: {self.source_h} x {self.source_w},\n"
            f"  target: {self.target_h} x {self.target_w},\n"
            f"  deform: {self.deform_h} x {self.deform_w}\n"
            f")"
        )