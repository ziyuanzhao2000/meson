import os
import csv
import heapq
import xml.etree.ElementTree as ET
from tqdm import tqdm
from collections import defaultdict
from typing import Optional, Sequence, Union


import numpy as np
import pandas as pd
from shapely.geometry import Polygon
import cv2
import tifffile 
import anndata as ad



def get_patch_scores(
    patch_table: Union[ad.AnnData, Sequence[ad.AnnData]], 
    feature_name: str
) -> np.ndarray:
    """
    Extract feature scores from a patch table or list of patch tables.
    
    Handles both sparse arrays (e.g., SAE scores stored in .X) and 
    dense arrays stored in .obs columns. When multiple tables are provided,
    scores are concatenated in order.
    
    Parameters
    ----------
    patch_table : AnnData or sequence of AnnData
        Patch-level data containing feature scores. Can be a single AnnData
        object or a list/sequence of AnnData objects.
    feature_name : str
        Name of the feature to extract.
        
    Returns
    -------
    scores : ndarray of shape (n_patches,)
        Feature scores as a 1D array. When multiple tables are provided,
        scores are concatenated in the order of input tables.
        
    Examples
    --------
    >>> # Single table
    >>> scores = get_patch_scores(patches, 'UNI_SAE_12345')
    
    >>> # Multiple tables
    >>> scores = get_patch_scores([patches1, patches2], 'UNI_SAE_12345')
    """
    # Handle list of tables
    if isinstance(patch_table, (list, tuple)):
        all_scores = []
        for table in patch_table:
            scores = _extract_scores_from_single_table(table, feature_name)
            all_scores.append(scores)
        return np.concatenate(all_scores)
    
    # Handle single table
    return _extract_scores_from_single_table(patch_table, feature_name)

def _extract_scores_from_single_table(patch_table: ad.AnnData, feature_name: str) -> np.ndarray:
    """Helper function to extract scores from a single AnnData table."""
    # Try to get from .X (sparse matrix)
    if feature_name in patch_table.var_names:
        scores = patch_table[:, feature_name].X.toarray()[:, 0]
    # Otherwise get from .obs (dense)
    elif feature_name in patch_table.obs.columns:
        scores = patch_table.obs[feature_name].to_numpy()
    else:
        raise KeyError(
            f"Feature '{feature_name}' not found in patch table. "
            f"Available in .var: {list(patch_table.var_names)}, "
            f"Available in .obs: {list(patch_table.obs.columns)}"
        )
    return scores


def copy_feature_score_to_obs(
    patch_table: ad.AnnData,
    feature_name: str,
    obs_colname: Optional[str] = None,
) -> ad.AnnData:
    """
    Copy one feature score from ``patch_table[:, feature_name].X`` into ``patch_table.obs``.

    Parameters
    ----------
    patch_table : AnnData
        Patch-level AnnData table.
    feature_name : str
        Feature name in ``patch_table.var_names``.
    obs_colname : str, optional
        Output column name in ``patch_table.obs``. Defaults to ``feature_name``.

    Returns
    -------
    patch_table : AnnData
        The same AnnData object, modified in place.
    """
    if feature_name not in patch_table.var_names:
        raise KeyError(
            f"Feature '{feature_name}' not found in patch_table.var_names."
        )

    if obs_colname is None:
        obs_colname = feature_name

    x = patch_table[:, feature_name].X
    if hasattr(x, "toarray"):
        scores = x.toarray()[:, 0]
    else:
        scores = np.asarray(x).reshape(-1)

    patch_table.obs[obs_colname] = scores
    return patch_table


def select_random_patches(
    sdata,
    patch_table_names: Union[str, Sequence[str]],
    n: int,
    random_state: Optional[int] = None,
) -> ad.AnnData:
    """
    Randomly sample patches across one or more patch tables.
    
    Parameters
    ----------
    sdata : SpatialData-like
        Container that stores patch tables addressable by name.
    patch_table_names : str or sequence of str
        One patch table name or multiple names. Sampling is performed across
        the union of all listed tables.
    n : int
        Number of patches to randomly sample.
    random_state : int, optional
        Random seed for reproducibility.
        
    Returns
    -------
    sampled_patches : AnnData
        AnnData containing randomly sampled patches from all input tables.
        Adds `.obs['_source_patch_table']`.
        
    Examples
    --------
    >>> # Sample 100 random patches from one table
    >>> random_patches = select_random_patches(
    ...     sdata, 'LSP24530_grid_point_patch', n=100, random_state=42
    ... )
    
    >>> # Sample 200 random patches across multiple tables
    >>> random_patches = select_random_patches(
    ...     sdata,
    ...     ['LSP24530_grid_point_patch', 'LSP24531_grid_point_patch'],
    ...     n=200,
    ...     random_state=42
    ... )
    """
    if isinstance(patch_table_names, str):
        table_names = [patch_table_names]
    else:
        table_names = list(patch_table_names)

    if len(table_names) == 0:
        raise ValueError("patch_table_names must contain at least one table name.")

    if n < 0:
        raise ValueError("n must be >= 0.")

    if n == 0:
        return sdata[table_names[0]][[]].copy()

    # Set random seed if provided
    if random_state is not None:
        np.random.seed(random_state)

    # Collect all patch indices
    all_candidates = []  # (table_name, row_idx)
    
    for table_name in table_names:
        patch_table = sdata[table_name]
        n_patches = len(patch_table)
        
        for row_idx in range(n_patches):
            all_candidates.append((table_name, row_idx))
    
    if len(all_candidates) == 0:
        return sdata[table_names[0]][[]].copy()
    
    # Randomly sample n patches
    n_available = len(all_candidates)
    n_to_sample = min(n, n_available)
    
    if n_available < n:
        print(f"Warning: Only {n_available} patches available, sampling all.")
    
    sampled_idx = np.random.choice(n_available, size=n_to_sample, replace=False)
    selected_candidates = [all_candidates[i] for i in sampled_idx]
    
    # Group by table
    selected_indices = defaultdict(list)
    
    for table_name, row_idx in selected_candidates:
        selected_indices[table_name].append(row_idx)
    
    # Build output
    subsets = []
    for table_name in table_names:
        idx_list = selected_indices.get(table_name, [])
        if len(idx_list) == 0:
            continue
        subset = sdata[table_name][np.asarray(idx_list, dtype=np.int64)].copy()
        subset.obs['_source_patch_table'] = table_name
        subsets.append(subset)

    if len(subsets) == 0:
        return sdata[table_names[0]][[]].copy()

    return ad.concat(subsets, join='outer', merge='same')


def select_patches_for_binary_feature(
    sdata,
    patch_table_names: Union[str, Sequence[str]],
    feature_name: str,
    n: Optional[int] = None,
    random_state: Optional[int] = None,
) -> ad.AnnData:
    """
    Sample patches where a binary feature is active (value == 1).
    
    Collects all patches where the feature column equals 1, then randomly
    samples n patches from this set.
    
    Parameters
    ----------
    sdata : SpatialData-like
        Container that stores patch tables addressable by name.
    patch_table_names : str or sequence of str
        One patch table name or multiple names. Selection is performed across
        the union of all listed tables.
    feature_name : str
        Name of the binary feature column in .obs (e.g., 'kmeans_label_0').
    n : int, optional
        Number of patches to sample. If None, returns all active patches.
    random_state : int, optional
        Random seed for reproducibility.
        
    Returns
    -------
    sampled_patches : AnnData
        AnnData containing sampled patches where feature == 1.
        Adds `.obs['_source_patch_table']`.
        
    Examples
    --------
    >>> # Sample 20 patches where feature is active
    >>> active_patches = select_patches_for_binary_feature(
    ...     sdata,
    ...     'LSP24530_grid_point_patch',
    ...     'kmeans_label_0',
    ...     n=20,
    ...     random_state=42
    ... )
    
    >>> # Get all active patches across multiple tables
    >>> all_active = select_patches_for_binary_feature(
    ...     sdata,
    ...     ['LSP24530_grid_point_patch', 'LSP24531_grid_point_patch'],
    ...     'UNI_SAE_12345',
    ...     n=None
    ... )
    """
    if isinstance(patch_table_names, str):
        table_names = [patch_table_names]
    else:
        table_names = list(patch_table_names)

    if len(table_names) == 0:
        raise ValueError("patch_table_names must contain at least one table name.")

    if n is not None and n < 0:
        raise ValueError("n must be >= 0 or None.")

    if n == 0:
        return sdata[table_names[0]][[]].copy()

    # Set random seed if provided
    if random_state is not None:
        np.random.seed(random_state)

    # Collect all active patches (feature == 1)
    all_active_candidates = []  # (table_name, row_idx)
    
    for table_name in table_names:
        patch_table = sdata[table_name]
        
        # Check if feature exists in .obs
        if feature_name not in patch_table.obs.columns:
            print(f"Warning: Feature '{feature_name}' not found in table '{table_name}', skipping.")
            continue
        
        # Get active indices (where feature == 1)
        active_mask = patch_table.obs[feature_name] == 1
        active_indices = np.where(active_mask)[0]
        
        for row_idx in active_indices:
            all_active_candidates.append((table_name, row_idx))
    
    if len(all_active_candidates) == 0:
        raise ValueError(
            f"No active patches found for feature '{feature_name}' "
            f"in tables: {table_names}"
        )
    
    # Sample patches
    n_available = len(all_active_candidates)
    
    if n is None:
        selected_candidates = all_active_candidates
    else:
        n_to_sample = min(n, n_available)
        
        if n_available < n:
            print(f"Warning: Only {n_available} active patches available, sampling all.")
        
        sampled_idx = np.random.choice(n_available, size=n_to_sample, replace=False)
        selected_candidates = [all_active_candidates[i] for i in sampled_idx]
    
    # Group by table
    selected_indices = defaultdict(list)
    
    for table_name, row_idx in selected_candidates:
        selected_indices[table_name].append(row_idx)
    
    # Build output
    subsets = []
    for table_name in table_names:
        idx_list = selected_indices.get(table_name, [])
        if len(idx_list) == 0:
            continue
        subset = sdata[table_name][np.asarray(idx_list, dtype=np.int64)].copy()
        subset.obs['_source_patch_table'] = table_name
        subsets.append(subset)

    if len(subsets) == 0:
        return sdata[table_names[0]][[]].copy()

    return ad.concat(subsets, join='outer', merge='same')

def select_top_patches(
    sdata,
    patch_table_names: Union[str, Sequence[str]],
    feature_name: str,
    n: Optional[int] = None,
    min_score: Optional[float] = None,
    take_every: Optional[int] = 1,
) -> ad.AnnData:
    """
    Select top-scoring patches for a feature across one or more patch tables.
    
    Parameters
    ----------
    sdata : SpatialData-like
        Container that stores patch tables addressable by name.
    patch_table_names : str or sequence of str
        One patch table name or multiple names. Selection is performed across
        the union of all listed tables.
    feature_name : str
        Name of the feature to rank patches by.
    n : int, optional
        Number of top patches to return. If None, returns all patches with score > min_score.
    min_score : float, optional
        Minimum score threshold. Only used when n is None. Default is 0.
    take_every : int, optional
        If provided, samples every Nth patch from the sorted results instead of taking
        the strict top N. If n is also provided, samples n patches evenly spaced.
        If n is None, take_every determines the stride through all eligible patches.
        
    Returns
    -------
    top_patches : AnnData
        AnnData containing selected patches from all input tables, globally sorted
        by score descending. Adds `.obs['_source_patch_table']`.
        
    Examples
    --------
    >>> # Get top 100 patches for a feature
    >>> top_patches = select_top_patches(
    ...     sdata, 'LSP24530_grid_point_patch', 'UNI_SAE_12345', n=100
    ... )
    
    >>> # Get all patches with positive scores
    >>> active_patches = select_top_patches(
    ...     sdata,
    ...     ['LSP24530_grid_point_patch', 'LSP24531_grid_point_patch'],
    ...     'UNI_SAE_12345',
    ...     n=None,
    ...     min_score=0
    ... )
    
    >>> # Sample 100 patches evenly spaced from positive patches
    >>> sampled_patches = select_top_patches(
    ...     sdata, 'LSP24530_grid_point_patch', 'UNI_SAE_12345', 
    ...     n=100, min_score=0, take_every=None
    ... )  # take_every computed automatically
    """
    if isinstance(patch_table_names, str):
        table_names = [patch_table_names]
    else:
        table_names = list(patch_table_names)

    if len(table_names) == 0:
        raise ValueError("patch_table_names must contain at least one table name.")

    if n is not None and n < 0:
        raise ValueError("n must be >= 0 or None.")

    if n == 0:
        return sdata[table_names[0]][[]].copy()

    # First, collect all eligible patches and their scores
    all_candidates = []  # (score, table_name, row_idx)
    
    if min_score is None:
        min_score = 0.0 if (n is None or take_every is not None) else float('-inf')
    
    for table_name in table_names:
        patch_table = sdata[table_name]
        try:
            scores = get_patch_scores(patch_table, feature_name)
        except KeyError as e:
            raise KeyError(f"{e} (table='{table_name}')") from e

        keep_idx = np.where(scores > min_score)[0]
        if keep_idx.size == 0:
            continue
            
        for row_idx in keep_idx:
            all_candidates.append((float(scores[row_idx]), table_name, row_idx))
    
    # Sort all candidates by score descending
    all_candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Apply sampling logic
    if take_every is not None:
        # Use the provided stride
        stride = take_every
    elif n is not None and len(all_candidates) > 0:
        # Compute stride to get n samples evenly spaced
        stride = max(1, len(all_candidates) // n)
    else:
        # No sampling, take all
        stride = 1
    
    # Select patches with stride
    if n is not None:
        selected_candidates = all_candidates[::stride][:n]
    else:
        selected_candidates = all_candidates[::stride]
    
    # Group by table
    selected_indices = defaultdict(list)
    selected_scores = defaultdict(list)
    
    for score, table_name, row_idx in selected_candidates:
        selected_indices[table_name].append(row_idx)
        selected_scores[table_name].append(score)
    
    # Build output
    subsets = []
    for table_name in table_names:
        idx_list = selected_indices.get(table_name, [])
        if len(idx_list) == 0:
            continue
        subset = sdata[table_name][np.asarray(idx_list, dtype=np.int64)].copy()
        subset.obs['_source_patch_table'] = table_name
        subset.obs['_selection_score'] = np.asarray(selected_scores[table_name], dtype=np.float32)
        subsets.append(subset)

    if len(subsets) == 0:
        return sdata[table_names[0]][[]].copy()

    out = ad.concat(subsets, join='outer', merge='same')
    order = np.argsort(-out.obs['_selection_score'].to_numpy())
    out = out[order].copy()
    out.obs.drop(columns=['_selection_score'], inplace=True)
    return out


def select_negative_patches(
    sdata,
    patch_table_names: Union[str, Sequence[str]],
    feature_name: str,
    n: Optional[int] = None,
    take_every: Optional[int] = None,
) -> ad.AnnData:
    """
    Select patches with zero scores for a feature across one or more patch tables.
    
    Parameters
    ----------
    sdata : SpatialData-like
        Container that stores patch tables addressable by name.
    patch_table_names : str or sequence of str
        One patch table name or multiple names. Selection is performed across
        the union of all listed tables.
    feature_name : str
        Name of the feature to filter by (selects patches where score == 0).
    n : int, optional
        Number of patches to return. If None, returns all zero-score patches.
    take_every : int, optional
        If provided, samples every Nth patch from the zero-score patches.
        If n is also provided and take_every is None, computes stride automatically
        to get n evenly-spaced samples.
        
    Returns
    -------
    negative_patches : AnnData
        AnnData containing selected zero-score patches. Adds `.obs['_source_patch_table']`.
        
    Examples
    --------
    >>> # Get 100 evenly-spaced negative patches
    >>> neg_patches = select_negative_patches(
    ...     sdata, 'LSP24530_grid_point_patch', 'UNI_SAE_12345', n=100
    ... )
    
    >>> # Get every 10th negative patch across multiple tables
    >>> neg_patches = select_negative_patches(
    ...     sdata,
    ...     ['LSP24530_grid_point_patch', 'LSP24531_grid_point_patch'],
    ...     'UNI_SAE_12345',
    ...     take_every=10
    ... )
    """
    if isinstance(patch_table_names, str):
        table_names = [patch_table_names]
    else:
        table_names = list(patch_table_names)

    if len(table_names) == 0:
        raise ValueError("patch_table_names must contain at least one table name.")

    if n is not None and n < 0:
        raise ValueError("n must be >= 0 or None.")

    if n == 0:
        return sdata[table_names[0]][[]].copy()

    # Collect all zero-score patches
    all_candidates = []  # (table_name, row_idx)
    
    for table_name in table_names:
        patch_table = sdata[table_name]
        try:
            scores = get_patch_scores(patch_table, feature_name)
        except KeyError as e:
            raise KeyError(f"{e} (table='{table_name}')") from e

        zero_idx = np.where(scores == 0)[0]
        for row_idx in zero_idx:
            all_candidates.append((table_name, row_idx))
    
    if len(all_candidates) == 0:
        return sdata[table_names[0]][[]].copy()
    
    # Apply sampling logic
    if take_every is not None:
        stride = take_every
    elif n is not None:
        stride = max(1, len(all_candidates) // n)
    else:
        stride = 1
    
    # Select patches with stride
    if n is not None:
        selected_candidates = all_candidates[::stride][:n]
    else:
        selected_candidates = all_candidates[::stride]
    
    # Group by table
    selected_indices = defaultdict(list)
    
    for table_name, row_idx in selected_candidates:
        selected_indices[table_name].append(row_idx)
    
    # Build output
    subsets = []
    for table_name in table_names:
        idx_list = selected_indices.get(table_name, [])
        if len(idx_list) == 0:
            continue
        subset = sdata[table_name][np.asarray(idx_list, dtype=np.int64)].copy()
        subset.obs['_source_patch_table'] = table_name
        subsets.append(subset)

    if len(subsets) == 0:
        return sdata[table_names[0]][[]].copy()

    return ad.concat(subsets, join='outer', merge='same')

# def get_optimal_chunk_size(image, patch_chunk_size=256, target_size=500*1024*1024):
#     n_channels = len(image['c'])
#     element_size = np.dtype(image.dtype).itemsize
#     patch_size = element_size * patch_chunk_size * patch_chunk_size * n_channels
#     num_patches = max(1, int(target_size/patch_size))
#     chunk_size = patch_chunk_size * int(num_patches**0.5)
#     print("new chunk size", [n_channels, chunk_size, chunk_size])
#     return [n_channels, chunk_size, chunk_size]

def get_optimal_chunk_size(image):
    # assumes C x W x H
    chunksize = image.data.chunksize
    return (chunksize[0], chunksize[1]*2, chunksize[2]*2)
    # return (3, 512, 512)

# a. write a backup copy of the data
# def overwrite_element(sdata, name, new_name='_temp'):
#     sdata[new_name] = sdata[name]
#     sdata.write_element(new_name)
#     # b. rewrite the original data
#     sdata.delete_element_from_disk(name)
#     sdata.write_element(name)
#     # c. remove the backup copy
#     del sdata[new_name]
#     sdata.delete_element_from_disk(new_name)

### Code written by Soheil (Soheil_RastgouTalemi@hms.harvard.edu)
def xml2csv(xml_file_path):
    # Define the CSV file header
    csv_header = ['ROI ID', 'Text', 'SizeX', 'SizeY', 'Points']

    # Define the XML namespace
    namespace = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}

    csv_file_path = xml_file_path.replace('.ome.xml', '_ROIs.csv')

    # Load the XML file
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing XML file {xml_file_path}: {e}")
        return 

    # Extract SizeX and SizeY values
    pixels = root.find('.//ome:Pixels', namespace)
    size_x = pixels.attrib.get('SizeX', '') if pixels is not None else ''
    size_y = pixels.attrib.get('SizeY', '') if pixels is not None else ''

    # Open the CSV file in write mode and create a CSV writer object
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(csv_header)

        # Find all ROI elements
        roi_elements = root.findall('.//ome:ROI', namespace)

        # Iterate over each ROI element
        for roi in roi_elements:
            roi_id = roi.attrib.get('ID', '')
            # Find the Polygon element within the ROI
            polygon = roi.find('.//ome:Polygon', namespace)
            polyline = roi.find('.//ome:Polyline', namespace)
            name = roi.attrib.get('Name', '')

            if polygon is not None:
                text = polygon.attrib.get('Text', '')
                points = polygon.attrib.get('Points', '')

                # Write the ROI ID, text, points, SizeX, and SizeY to the CSV file
                writer.writerow([roi_id, text, size_x, size_y, points])

            if polyline is not None:
                text = name
                points = polyline.attrib.get('Points', '')

                # Write the ROI ID, text, points, SizeX, and SizeY to the CSV file
                writer.writerow([roi_id, text,  size_x, size_y, points])

def gdf2mask(gdf, size_x=None, size_y=None, 
                polys_colname='geometry',
                labels_colname=None,
                label_strategy='binary',
                verbose=False):
    return df2mask(gdf, size_x=size_x, size_y=size_y,
                   points_colname=polys_colname,
                   labels_colname=labels_colname,
                   label_strategy=label_strategy,
                   verbose=verbose,
                   is_geodataframe=True)
    
def df2mask(df, size_x=None, size_y=None, 
                points_colname='Points',
                labels_colname=None,
                label_strategy='binary',
                verbose=False,
                is_geodataframe=False):
    if size_x is None or size_y is None:
        size_x = int(df['SizeX'].iloc[0])
        size_y = int(df['SizeY'].iloc[0])
    if label_strategy == 'set':
        labels = set(df[labels_colname])
        indexed_sorted = {elem: i for i, elem in enumerate(sorted(labels), start=1)}

    img = np.zeros((size_y, size_x), dtype=np.int32)
    if len(df) > 0:
        iter_obj = tqdm(df.iterrows(), total=len(df)) if verbose else df.iterrows()
        for _, row in iter_obj:
            if is_geodataframe:
                polygon = row[points_colname]
            else:
                polygon = points2poly(row[points_colname])
            poly_points = np.array(polygon.exterior.coords).astype(np.int32)
            if label_strategy == 'binary':
                color = 1
            elif label_strategy == 'set':
                color = indexed_sorted.get(row[labels_colname], 0)
            elif label_strategy == 'index':
                color = int(row[labels_colname])
            cv2.fillPoly(img, [poly_points], color=color)
    return img

### Code written by Soheil (Soheil_RastgouTalemi@hms.harvard.edu)
### Adapted by Ziyuan on 08/26/2025
def csv2mask(csv_file_path, 
             verbose=False, 
             compression='zstd'):
    ROIS = pd.read_csv(csv_file_path)

    # Ensure 'SizeX' and 'SizeY' columns exist
    if 'SizeX' not in ROIS.columns or 'SizeY' not in ROIS.columns:
        print(f"Skipping {csv_file_path}: 'SizeX' or 'SizeY' columns are missing.")
        return

    img = df2mask(ROIS, verbose=verbose)

    mask_file_path = os.path.splitext(csv_file_path)[0] + '_mask.ome.tiff'
    tifffile.imwrite(mask_file_path, 
                     img,
                     photometric='minisblack',
                     metadata={
                         'axes': 'YX'
                     },
                     tile=(1024, 1024),
                     compression=compression,
                     ome=True,
                     bigtiff=True)
    print(f"Saved mask to {mask_file_path}")


def points2poly(points):
    points = points.split()
    xy_coordinates = [point.split(',') for point in points]
    xy_array = np.array(xy_coordinates).astype(np.float64)
    return Polygon(xy_array)