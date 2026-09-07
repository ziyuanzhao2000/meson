import geopandas as gpd
import numpy as np
import spatialdata as sd
from spatialdata.transformations import Identity, set_transformation
from shapely import Polygon

def make_bbox(sdata, 
              image_name: str,
              size, 
              *,
              points = None,
              point_name = 'grid_point',  
              shape_name = 'bbox',
              cs: str | None = None):
    if points is None:
        points = sdata[f'{image_name}_{point_name}']
    if cs is None:
        cs = image_name.split('_')[0]
    points = points.compute().to_numpy()
    half_size = size / 2
    offsets = np.array([
        [-half_size, -half_size],  # bottom-left
        [half_size, -half_size],   # bottom-right
        [half_size, half_size],    # top-right
        [-half_size, half_size]    # top-left
    ])
    
    corners = points[:, np.newaxis, :] + offsets
    boxes = [Polygon(corner) for corner in corners]
    
    gdf = gpd.GeoDataFrame(geometry=boxes)
    shapes = sd.models.ShapesModel.parse(gdf)
    set_transformation(shapes, Identity(), to_coordinate_system=cs)
    sdata[f'{image_name}_{point_name}_{shape_name}'] = shapes
    
    return sdata
