import dask.array as da
import anndata as ad
import pandas as pd
import xarray as xr
import numpy as np
from spatialdata import SpatialData
from spatialdata.models import TableModel, PointsModel
from ._make_bbox import make_bbox

def _filter_points(sdata, image_name, point_name, size):
    points = sdata[f'{image_name}_{point_name}'].compute()
    tissue_mask = sdata[f'{image_name}_tissue']

    height, width = tissue_mask.shape
    half_size = size // 2
    
    y_coords = points['y'].to_numpy().astype(int)
    x_coords = points['x'].to_numpy().astype(int)

    boundary_mask = (
        (x_coords >= half_size) & 
        (x_coords < width - half_size) &
        (y_coords >= half_size) & 
        (y_coords < height - half_size)
    )
    filtered_points = points[boundary_mask]
    y_coords = xr.DataArray(y_coords[boundary_mask])
    x_coords = xr.DataArray(x_coords[boundary_mask])
    mask_values = tissue_mask.sel(y=y_coords, x=x_coords, method="nearest").astype(bool)
    return PointsModel.parse(filtered_points[np.bool(mask_values)])

def make_patch(sdata: SpatialData, 
               image_name: str,
               size: int, 
               *,
               point_name: str = 'grid_point',
               annotation_name: str = 'patch',
               cs: str | None = None):
    if cs is None:
        cs = image_name.split('_')[0]

    points = _filter_points(sdata, image_name, point_name, size)
    sdata = make_bbox(sdata, image_name=image_name, point_name=point_name, 
                      size=100, shape_name='bbox', cs=cs)
    shape_name = f"{image_name}_{point_name}_bbox"
    image = sdata[image_name]
    print(image_name, image.shape)
    half_size = size // 2
    _, height, width = image.shape
    valid_points = []
    valid_patches = []

    for idx, point in points.iterrows():
        x, y = point.astype(int)
        # Check if patch would be fully within image bounds
        if (x >= half_size and x < width - half_size and 
            y >= half_size and y < height - half_size):
            valid_points.append(idx)
            patch = image[
                :,  # All channels
                y - half_size:y + half_size,
                x - half_size:x + half_size
            ]
            valid_patches.append(patch)

    patches_array = da.stack(valid_patches)
    obs = pd.DataFrame()
    obs["instance_id"] = valid_points
    obs["region"] = shape_name
    obs["image"] = image_name
    obs["is_tissue"] = True
    obs["region"].astype("category")
    adata = ad.AnnData(
        X=np.zeros((len(valid_points), 1)),  # Empty data matrix
        # X = np.array(()),
        obs=obs,
        obsm={'patch': patches_array}  # Store patches in obsm
    )
    table = TableModel.parse(adata, 
                             region=shape_name, 
                             region_key="region", 
                             instance_key="instance_id")
    sdata[f'{image_name}_{point_name}_{annotation_name}'] = table
    return sdata