import numpy as np
from spatialdata.models import PointsModel
from spatialdata.transformations import set_transformation, Identity
from xarray.core.datatree import DataTree
from typing import Tuple
from meson._readwrite import get_base_level

def make_grid(sdata, 
              image_name: str, 
              step: Tuple[int, int], 
              *,
              shift: Tuple[int, int] = (0, 0), 
              point_name: str = 'grid_point',
              cs: str | None = None):
    img = get_base_level(sdata[image_name])
    if cs is None:
        cs = image_name.split('_')[0]
    # generalizes to 3D: https://stackoverflow.com/questions/12864445/how-to-convert-the-output-of-meshgrid-to-the-corresponding-array-of-points
    min_x, min_y = shift # assuming 2D image
    max_y, max_x = img.shape[1:]
    step_x, step_y = step
    min_x, min_y = min_x + step_x//2, min_y + step_y//2
    grid_x, grid_y = np.mgrid[min_x:max_x:step_x, min_y:max_y:step_y]
    coords = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
    points = PointsModel.parse(coords)
    set_transformation(element=points, 
                       transformation=Identity(), 
                       to_coordinate_system=cs)
    sdata[f'{image_name}_{point_name}'] = points
    return sdata 