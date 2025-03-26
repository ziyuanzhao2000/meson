import dask.array as da
import anndata as ad
import pandas as pd
import xarray as xr
import numpy as np
from spatialdata import SpatialData, bounding_box_query, polygon_query, to_polygons
from spatialdata.models import TableModel, PointsModel
from spatialdata.transformations import set_transformation, Affine, Identity
from ._make_bbox import make_bbox
from .._readwrite import get_base_level, get_scaling_factor
from .._utils import get_optimal_chunk_size
from shapely.affinity import affine_transform
from tqdm import tqdm

def _filter_points(sdata, image_name, point_name, size):
    img_obj = sdata[image_name]
    cs = image_name.split('_')[0]
    points = sdata[f'{image_name}_{point_name}']
    tissue_mask = sdata[f'{image_name}_tissue']
    mask_polygon = to_polygons(tissue_mask)
    scaling_factors = get_scaling_factor(img_obj)
    affine = Affine(np.eye(3) * [*scaling_factors, 1], 
                    input_axes=['x', 'y'], 
                    output_axes=['x', 'y'])
    set_transformation(mask_polygon, affine, cs)
    filtered_points = points.copy()
    filtered_points['ID'] = points.index
    for polygon in mask_polygon['geometry']:
        transformed_polygon = affine_transform(polygon, [scaling_factors[0], 0, 0, scaling_factors[1], 0, 0])
        filtered_points = polygon_query(filtered_points, 
                                        polygon=transformed_polygon,
                                        target_coordinate_system=cs)
    xmin, ymin = size/2, size/2
    ymax, xmax = np.array(get_base_level(img_obj).shape[1:]) - size/2
    filtered_points = PointsModel.parse(filtered_points.compute())
    set_transformation(filtered_points, Identity(), cs)
    filtered_points = bounding_box_query(filtered_points, 
                                         axes=['x', 'y'],
                                         min_coordinate=[xmin, ymin], 
                                         max_coordinate=[xmax, ymax], 
                                         target_coordinate_system=cs)
    filtered_points = filtered_points.set_index('ID')
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
                      size=size, shape_name='bbox', cs=cs)
    shape_name = f"{image_name}_{point_name}_bbox"
    # image = get_base_level(sdata[image_name])
    # image = image.chunk(chunks=get_optimal_chunk_size(image)) # via xarray
    # # might want to reset this
    half_size = size // 2
    # patches = []

    # for _, point in tqdm(points.iterrows()):
    #     x, y = point.astype(int)
    #     patch = image[
    #         :,  # All channels
    #         y - half_size:y + half_size,
    #         x - half_size:x + half_size
    #     ]
    #     patches.append(patch)
    # patches_array = da.stack(patches, axis=0)

    obs = pd.DataFrame()
    obs["instance_id"] = points.index
    obs["region"] = shape_name
    obs["region"].astype("category")
    obs["image"] = image_name
    obs["is_tissue"] = True
    obs["ymin"] = (points['y'] - half_size).astype(int)
    obs["ymax"] = (points['y'] + half_size).astype(int)
    obs["xmin"] = (points['x'] - half_size).astype(int)
    obs["xmax"] = (points['x'] + half_size).astype(int)

    adata = ad.AnnData(
        X=np.zeros((len(points), 1)),  # Empty data matrix
        obs=obs,
        # obsm={'patch': patches_array}  # Store patches in obsm
    )
    table = TableModel.parse(adata, 
                             region=shape_name, 
                             region_key="region", 
                             instance_key="instance_id")
    sdata[f'{image_name}_{point_name}_{annotation_name}'] = table
    return sdata