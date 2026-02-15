import os
import csv
import xml.etree.ElementTree as ET
from tqdm import tqdm
from typing import Optional, Union


import numpy as np
import pandas as pd
from shapely.geometry import Polygon
import cv2
import tifffile 
import anndata as ad

def get_patch_scores(patch_table: ad.AnnData, feature_name: str) -> np.ndarray:
    """
    Extract feature scores from a patch table.
    
    Handles both sparse arrays (e.g., SAE scores stored in .X) and 
    dense arrays stored in .obs columns.
    
    Parameters
    ----------
    patch_table : AnnData
        Patch-level data containing feature scores.
    feature_name : str
        Name of the feature to extract.
        
    Returns
    -------
    scores : ndarray of shape (n_patches,)
        Feature scores as a 1D array.
        
    Examples
    --------
    >>> scores = get_patch_scores(patches, 'UNI_SAE_12345')
    """
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

def select_top_patches(
    patch_table: ad.AnnData, 
    feature_name: str, 
    n: Optional[int] = None,
    min_score: Optional[float] = None
) -> ad.AnnData:
    """
    Select top-scoring patches for a feature from a patch table.
    
    Parameters
    ----------
    patch_table : AnnData
        Patch-level data with bounding boxes in .obs (xmin, xmax, ymin, ymax)
        and feature scores in .X or .obs.
    feature_name : str
        Name of the feature to rank patches by.
    n : int, optional
        Number of top patches to return. If None, returns all patches with score > min_score.
    min_score : float, optional
        Minimum score threshold. Only used when n is None. Default is 0.
        
    Returns
    -------
    top_patches : AnnData
        Subset of patch_table containing the top N (or all above threshold) patches,
        sorted by score in descending order.
        
    Examples
    --------
    >>> # Get top 100 patches for a feature
    >>> top_patches = select_top_patches(patches, 'UNI_SAE_12345', n=100)
    
    >>> # Get all patches with positive scores
    >>> active_patches = select_top_patches(patches, 'UNI_SAE_12345', n=None, min_score=0)
    """
    # Extract scores
    scores = get_patch_scores(patch_table, feature_name)
    
    # Select indices
    if n is not None:
        # Top N patches
        sort_idx = np.argsort(-scores)[:n]
    else:
        # All patches above threshold
        if min_score is None:
            min_score = 0.0
        keep_idx = np.where(scores > min_score)[0]
        sort_idx = keep_idx[np.argsort(-scores[keep_idx])]
    
    # Return subset
    return patch_table[sort_idx].copy()

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