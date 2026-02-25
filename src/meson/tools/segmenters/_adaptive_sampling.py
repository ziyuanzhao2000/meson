"""
Adaptive quadtree sampling for whole slide image segmentation.
"""
import numpy as np
from meson.tools._embed_patch import embed_patch
import torch
import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Union
from dataclasses import dataclass
from collections import deque
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import griddata
from skimage.filters import sobel
import heapq

from meson._readwrite import get_base_level


@dataclass
class Cell:
    """
    Represents a cell in the quadtree subdivision.
    
    Attributes
    ----------
    corners : list
        List of (x, y) tuples representing the four corners of the cell.
        Order: [top_left, top_right, bottom_left, bottom_right]
    size : int
        The size (width/height) of the square cell in pixels
    """
    corners: list
    size: int


def initialize_grid(
    patch_df,
    size: int,
    patch_size: int = 448,
    feature_column: Optional[str] = None
) -> Tuple[Dict[Tuple[int, int], int], deque]:
    """
    Initialize sampling grid from patch table.
    
    Parameters
    ----------
    patch_df : pd.DataFrame
        Patch table with xmin, ymin, xmax, ymax columns
    size : int
        Initial cell size for quadtree
    patch_size : int, default=448
        Size of image patches
    feature_column : str, optional
        Column name containing initial cluster labels. If None, all samples
        are initialized as background (-1)
        
    Returns
    -------
    samples : Dict[Tuple[int, int], int]
        Dictionary mapping (x, y) coordinates to cluster labels
    cells : deque
        Queue of cells to process
    """
    samples = {}
    cells = deque()
    
    for xmin in patch_df.xmin.unique():
        for ymin in patch_df.ymin.unique():
            xcenter = xmin + patch_size // 2
            ycenter = ymin + patch_size // 2
            samples[(xcenter, ycenter)] = -1  # Initialize as background
            cells.append(Cell(
                corners=[
                    (xcenter, ycenter),
                    (xcenter + size, ycenter),
                    (xcenter, ycenter + size),
                    (xcenter + size, ycenter + size)
                ],
                size=size
            ))
    
    # Set initial labels if provided
    if feature_column is not None:
        for row in patch_df.itertuples():
            xcenter = row.xmin + patch_size // 2
            ycenter = row.ymin + patch_size // 2
            samples[(xcenter, ycenter)] = getattr(row, feature_column)
    
    return samples, cells


def interpolate_edt(
    samples: Dict[Tuple[int, int], int],
    height: int,
    width: int,
    cluster_idx: Union[int, List[int]],
    downsample: int = 1,
    upsample: bool = True
) -> np.ndarray:
    """
    Interpolate binary mask using Euclidean Distance Transform.
    
    Parameters
    ----------
    samples : Dict[Tuple[int, int], int]
        Dictionary mapping (x, y) coordinates to cluster labels
    height : int
        Height of output image
    width : int
        Width of output image
    cluster_idx : int or list of int
        Target cluster index(es) for foreground
    downsample : int, default=1
        Factor by which to downsample before interpolation
    upsample : bool, default=True
        Whether to upsample result back to original resolution
        
    Returns
    -------
    interpolated : np.ndarray
        Binary mask with interpolated values
    """
    if isinstance(cluster_idx, int):
        cluster_idx = [cluster_idx]
    
    full_height, full_width = height, width
    height, width = height // downsample, width // downsample
    
    canvas_foreground = np.zeros((height, width), dtype=bool)
    canvas_background = np.zeros((height, width), dtype=bool)
    
    for (x, y), value in samples.items():
        x, y = x // downsample, y // downsample
        x = np.clip(x, 0, width - 1)
        y = np.clip(y, 0, height - 1)
        if value in cluster_idx:
            canvas_foreground[y, x] = True
        else:
            canvas_background[y, x] = True
    
    edt_foreground = distance_transform_edt(~canvas_background)
    edt_background = distance_transform_edt(~canvas_foreground)
    interpolated = edt_foreground > edt_background
    
    if downsample > 1 and upsample:
        import cv2
        interpolated = cv2.resize(
            interpolated.astype(np.uint8),
            (full_width, full_height),
            interpolation=cv2.INTER_NEAREST_EXACT
        ).astype(bool)
    
    return interpolated


def interpolate_linear(
    samples: Dict[Tuple[int, int], int],
    height: int,
    width: int,
    downsample: int = 1,
    mapping: Optional[Dict[int, int]] = None
) -> np.ndarray:
    """
    Interpolate continuous values using linear griddata interpolation.
    
    Parameters
    ----------
    samples : Dict[Tuple[int, int], int]
        Dictionary mapping (x, y) coordinates to cluster labels
    height : int
        Height of output image
    width : int
        Width of output image
    downsample : int, default=1
        Factor by which to downsample the output
    mapping : dict, optional
        Dictionary to remap cluster labels before interpolation
        
    Returns
    -------
    interpolated : np.ndarray
        Continuously interpolated values in downsampled resolution
    """
    # Convert samples to arrays
    points = np.array(list(samples.keys()), dtype=float)
    values = list(samples.values())
    
    if mapping is not None:
        values = [mapping[v] for v in values]
    values = np.array(values, dtype=float)
    
    # Create downsampled grid
    out_height = height // downsample
    out_width = width // downsample
    
    # Downsample point coordinates and swap to (y, x) order
    points_down = points / downsample
    points_down = points_down[:, [1, 0]]  # (x, y) -> (y, x)
    
    # Create meshgrid for interpolation
    grid_y, grid_x = np.mgrid[0:out_height, 0:out_width]
    
    # Interpolate using linear method
    interpolated = griddata(
        points_down, values, (grid_y, grid_x),
        method='linear', fill_value=0.0
    )
    
    return interpolated


def adaptive_refine_step(
    samples: Dict[Tuple[int, int], int],
    cells: deque,
    image: np.ndarray,
    embedder,
    classifier,
    batch_size: int,
    size_threshold: int = 4,
    min_gradient_threshold: float = 1e-8,
    mapping: Optional[Dict[int, int]] = None,
    show_progress: bool = False
) -> Tuple[Dict, deque, int, np.ndarray]:
    """
    Perform one iteration of gradient-based adaptive quadtree refinement.
    
    Parameters
    ----------
    samples : Dict[Tuple[int, int], int]
        Current samples dictionary
    cells : deque
        Queue of cells to process
    image : np.ndarray
        Full WSI image (C, H, W)
    embedder : ModelManager
        Embedding model (e.g., UNIEmbedder)
    classifier : FrequencyRankedKMeans
        Fitted clustering model
    batch_size : int
        Maximum number of new samples per iteration
    size_threshold : int, default=4
        Minimum cell size for subdivision
    min_gradient_threshold : float, default=1e-8
        Minimum gradient to consider for refinement
    mapping : dict, optional
        Mapping to apply to cluster labels for gradient computation
    show_progress : bool, default=False
        Whether to show progress bars
        
    Returns
    -------
    samples : Dict[Tuple[int, int], int]
        Updated samples dictionary
    cells : deque
        Updated cell queue
    n_samples : int
        Number of samples added
    gradient_map : np.ndarray
        Computed gradient map
    """
    if not cells:
        return samples, cells, 0, None
    
    _, height, width = image.shape
    
    # Rasterize current samples with linear interpolation
    interpolated = interpolate_linear(
        samples, height, width,
        downsample=size_threshold,
        mapping=mapping
    )
    
    # Compute gradient map
    gradient_map = sobel(interpolated)
    grad_height, grad_width = gradient_map.shape
    
    # Build priority queue based on gradient
    cell_heap = []
    counter = 0
    
    for cell in cells:
        top_left, top_right, bottom_left, bottom_right = cell.corners
        x_min_down = top_left[0] // size_threshold
        x_max_down = top_right[0] // size_threshold
        y_min_down = top_left[1] // size_threshold
        y_max_down = bottom_left[1] // size_threshold
        
        # Clip to valid gradient map range
        x_min_down = max(0, min(x_min_down, grad_width - 1))
        x_max_down = max(0, min(x_max_down, grad_width - 1))
        y_min_down = max(0, min(y_min_down, grad_height - 1))
        y_max_down = max(0, min(y_max_down, grad_height - 1))
        
        # Compute average gradient
        if x_max_down > x_min_down and y_max_down > y_min_down:
            gradient_region = gradient_map[
                y_min_down:y_max_down+1,
                x_min_down:x_max_down+1
            ]
            avg_gradient = np.mean(gradient_region)
        else:
            avg_gradient = gradient_map[y_min_down, x_min_down]
        
        heapq.heappush(cell_heap, (-avg_gradient, counter, cell))
        counter += 1
    
    # Collect cells to process
    coords_to_sample = set()
    cells_to_split = []
    
    while cell_heap and len(coords_to_sample) < batch_size:
        neg_gradient, _, cell = heapq.heappop(cell_heap)
        
        if neg_gradient > -min_gradient_threshold:
            break
        
        # Collect unsampled corners
        cell_coords_needed = [
            corner for corner in cell.corners
            if corner not in samples
        ]
        
        if len(coords_to_sample) + len(cell_coords_needed) > batch_size and coords_to_sample:
            heapq.heappush(cell_heap, (neg_gradient, counter, cell))
            counter += 1
            break
        
        coords_to_sample.update(cell_coords_needed)
        
        if cell.size > size_threshold:
            cells_to_split.append(cell)
    
    # Batch sample all required coordinates
    coords_list = list(coords_to_sample)
    n_samples = len(coords_list)
    
    if n_samples > 0:
        # Extract patches
        batch_cpu = torch.empty((n_samples, 3, 448, 448), dtype=torch.uint8)
        img_height, img_width = image.shape[1], image.shape[2]
        
        iterator = enumerate(coords_list)
        if show_progress:
            iterator = tqdm(iterator, total=n_samples, desc="Extracting patches")
        
        for idx, (x, y) in iterator:
            y_start, y_end = y - 224, y + 224
            x_start, x_end = x - 224, x + 224
            
            patch_np = image[
                :,
                max(0, y_start):min(img_height, y_end),
                max(0, x_start):min(img_width, x_end)
            ]
            
            # Pad if needed
            pad_top = max(0, -y_start)
            pad_bottom = max(0, y_end - img_height)
            pad_left = max(0, -x_start)
            pad_right = max(0, x_end - img_width)
            
            if any([pad_top, pad_bottom, pad_left, pad_right]):
                patch_np = np.pad(
                    patch_np,
                    ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
                    mode='constant', constant_values=0
                )
            
            batch_cpu[idx] = torch.from_numpy(patch_np)
        
        # Get embeddings and predict
        with torch.no_grad():
            embeddings = embedder(batch_cpu).cpu().numpy().astype(np.float64)
        values = classifier.predict(embeddings)
        
        # Update samples
        for coord, value in zip(coords_list, values):
            samples[coord] = value
    
    # Split cells and create children
    new_cells = deque()
    
    for cell in cells_to_split:
        top_left, top_right, bottom_left, bottom_right = cell.corners
        size = cell.size
        
        mid_x = top_left[0] + size // 2
        mid_y = top_left[1] + size // 2
        
        # Create 4 quadtree child cells
        new_cells.append(Cell(
            corners=[
                top_left,
                (mid_x, top_left[1]),
                (top_left[0], mid_y),
                (mid_x, mid_y)
            ],
            size=size // 2
        ))
        new_cells.append(Cell(
            corners=[
                (mid_x, top_left[1]),
                top_right,
                (mid_x, mid_y),
                (top_right[0], mid_y)
            ],
            size=size // 2
        ))
        new_cells.append(Cell(
            corners=[
                (top_left[0], mid_y),
                (mid_x, mid_y),
                bottom_left,
                (mid_x, bottom_left[1])
            ],
            size=size // 2
        ))
        new_cells.append(Cell(
            corners=[
                (mid_x, mid_y),
                (top_right[0], mid_y),
                (mid_x, bottom_left[1]),
                bottom_right
            ],
            size=size // 2
        ))
    
    # Combine remaining and new cells
    remaining_cells = deque()
    while cell_heap:
        _, _, cell = heapq.heappop(cell_heap)
        remaining_cells.append(cell)
    
    remaining_cells.extend(new_cells)
    
    return samples, remaining_cells, n_samples, gradient_map


import pandas as pd
import pyarrow.parquet as pq

def save_samples_parquet(
    samples: Dict[Tuple[int, int], int],
    path: Union[str, Path],
    compress: str = 'zstd'
) -> None:
    """
    Save samples to Parquet format (most compact and efficient).
    
    Parameters
    ----------
    samples : Dict[Tuple[int, int], int]
        Dictionary mapping (x, y) coordinates to cluster labels
    path : str or Path
        Output path for parquet file
    compress : str, default='zstd'
        Compression algorithm ('zstd', 'gzip', 'snappy', 'brotli')
    """
    df = pd.DataFrame([
        {'x': x, 'y': y, 'label': label}
        for (x, y), label in samples.items()
    ])
    df = df.astype({'x': 'int32', 'y': 'int32', 'label': 'int8'})
    df.to_parquet(path, compression=compress, index=False)


def load_samples_parquet(path: Union[str, Path]) -> Dict[Tuple[int, int], int]:
    """Load samples from Parquet format."""
    df = pd.read_parquet(path)
    return {(row.x, row.y): row.label for row in df.itertuples(index=False)}


def adaptive_sample_wsi(
    sdata,
    image_name: str,
    embedder,
    classifier,
    cluster_groups: List[List[int]],
    output_dir: Union[str, Path],
    initial_size: int = 256,
    batch_size: int = 1024,
    size_threshold: int = 32,
    patch_size: int = 448,
    point_name: str = 'grid_point',
    save_format: str = 'parquet', 
    show_progress: bool = True
) -> Dict[Tuple[int, int], int]:
    """
    Perform adaptive quadtree sampling on a whole slide image.
    
    Parameters
    ----------
    sdata : SpatialData
        Spatial data object containing the WSI
    image_name : str
        Name of the image to process
    embedder : ModelManager
        Embedding model (e.g., UNIEmbedder)
    classifier : FrequencyRankedKMeans
        Fitted clustering model
    cluster_groups : list of list of int
        Groups of cluster indices to refine sequentially
    output_dir : str or Path
        Directory to save results
    initial_size : int, default=256
        Initial cell size for quadtree
    batch_size : int, default=1024
        Batch size for sampling
    size_threshold : int, default=32
        Minimum cell size for subdivision
    patch_size : int, default=448
        Size of image patches
    point_name : str, default='grid_point'
        Name of the point element in spatial data
    save_format : str, default='parquet'
        Output format: 'parquet' (recommended), 'npz', 'hdf5', 'zarr', or 'pickle'
    show_progress : bool, default=True
        Whether to show progress bars
        
    Returns
    -------
    samples : Dict[Tuple[int, int], int]
        Final samples dictionary mapping coordinates to cluster labels
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get k means labels
    sdata.attrs['models'][f'kmeans'] = classifier
    sdata.attrs['models_metadata'].append({
        'name': f'kmeans',
        'model_type': 'KMeans'
    })
    print(f"Embedding patches with KMeans...", flush=True)
    sdata = embed_patch(
        sdata,
        embedder=f'kmeans',  # Model name from sdata.attrs['models']
        image_name=image_name,
        point_name='grid_point',
        patch_name='patch',
        obsm_key='UNI_embedding',  # Base embeddings to cluster
        save=False,
        overwrite=False
    )
    
    
    # Get patch table and WSI
    print("Getting patch table and WSI...", flush=True)
    patch_table = sdata[f'{image_name}_{point_name}_patch']
    wsi = get_base_level(sdata[image_name]).compute().data
    
    # Initialize grid
    print("Initializing grid...", flush=True)
    samples, cells = initialize_grid(
        patch_table.obs,
        size=initial_size,
        patch_size=patch_size,
        feature_column='kmeans_label'
    )
    
    if show_progress:
        print(f"Initialized with {len(samples)} samples and {len(cells)} cells.", flush=True)
    
    # Refine for each cluster group
    for cluster_idxs in cluster_groups:
        if show_progress:
            print(f"\nRefining for cluster indices: {cluster_idxs}", flush=True)
        
        # Reset cells for this group
        _, cells = initialize_grid(
            patch_table.obs,
            size=initial_size,
            patch_size=patch_size,
            feature_column=None
        )
        
        # Create mapping: target clusters -> 1, background -> 0
        mapping = {i: 0 for i in range(-1, 26)}
        for cluster_idx in cluster_idxs:
            mapping[cluster_idx] = 1
        
        total_samples = len(samples)
        n_iters = 0
        
        while cells:
            n_iters += 1
            samples, cells, n_new_samples, _ = adaptive_refine_step(
                samples, cells, wsi,
                embedder, classifier,
                batch_size=batch_size,
                size_threshold=size_threshold,
                mapping=mapping,
                show_progress=show_progress
            )
            total_samples += n_new_samples
            if n_new_samples == 0 and n_iters > 1:
                print("No new samples added, stopping refinement for this cluster group.", flush=True)
                break
            
            if show_progress:
                print(f"  Iteration {n_iters}: {len(cells)} cells remaining, "
                      f"{total_samples} samples total.", flush=True)
        
        if show_progress:
            print(f"  Refinement completed in {n_iters} iterations.", flush=True)
    
    # Save results
    if save_format == 'parquet':
        samples_path = output_dir / f'{image_name}_samples.parquet'
        save_samples_parquet(samples, samples_path)
    else:
        samples_path = output_dir / f'{image_name}_samples.pkl'
        with open(samples_path, 'wb') as f:
            pickle.dump(samples, f)
    
    if show_progress:
        print(f"\nSaved samples dictionary to: {samples_path}", flush=True)
    
    return samples