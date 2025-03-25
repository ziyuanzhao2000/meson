import dask.array as da
import anndata as ad
import pandas as pd
import xarray as xr
import numpy as np
from spatialdata import SpatialData, bounding_box_query, polygon_query, to_polygons
from spatialdata.models import TableModel, PointsModel
from spatialdata.transformations import set_transformation, Affine
from ._make_bbox import make_bbox
from .._readwrite import get_base_level, get_scaling_factor
from shapely.affinity import affine_transform

def _filter_points(sdata, image_name, point_name, size):
    img_obj = sdata[image_name]
    points = sdata[f'{image_name}_{point_name}']
    tissue_mask = sdata[f'{image_name}_tissue']
    size = np.array(size)
    xmin, ymin = size/2
    ymax, xmax = get_base_level(img_obj).shape[1:] - size/2
    print(len(points))
    filtered_points = bounding_box_query(points, 
                                         axes=['x', 'y'],
                                         min_coordinate=[xmin, ymin], 
                                         max_coordinate=[xmax, ymax], 
                                         target_coordinate_system=image_name)
    print(len(filtered_points))
    mask_polygon = to_polygons(tissue_mask)
    scaling_factors = get_scaling_factor(img_obj)
    affine = Affine(np.eye(3) * [*scaling_factors, 1], 
                    input_axes=['x', 'y'], output_axes=['x', 'y'])
    set_transformation(mask_polygon, affine, image_name)
    for polygon in mask_polygon['geometry']:
        transformed_polygon = affine_transform(polygon, [scaling_factors[0], 0, 0, scaling_factors[1], 0, 0])
        filtered_points = polygon_query(filtered_points, 
                                        polygon=transformed_polygon,
                                        target_coordinate_system=image_name)
    print(len(filtered_points))
    return filtered_points


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