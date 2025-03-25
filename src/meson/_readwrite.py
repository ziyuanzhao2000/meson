import openslide
import spatialdata as sd
import dask.array as da
import numpy as np
import copy

from xarray.core.dataarray import DataArray
from xarray.core.datatree import DataTree
from ._settings import settings
from spatialdata._io.io_raster import _read_multiscale
from spatialdata.models import Image2DModel
from tqdm import tqdm
from ._utils import get_optimal_chunk_size
import pandas as pd

img_exts = {
    "tif",
    "tiff",
    "ome.tif",
    "ome.tiff",
    "zarr"
}

def read(

    ):
    filekey = str(filename)
    filename = settings.writedir / (filekey + "." + settings.file_format_data)
    if not filename.exists():
        msg = (
            f"Reading with filekey {filekey!r} failed, "
            f"the inferred filename {filename!r} does not exist. "
            "If you intended to provide a filename, either use a filename "
            f"ending on one of the available extensions {img_exts} "
            "or pass the parameter `ext`."
        )
        raise ValueError(msg)

def read_HnE():
    pass

def read_mIF():
    pass

def get_base_level(image: DataTree | DataArray):
    if isinstance(image, DataTree):
        return sd.get_pyramid_levels(image, n=0)
    else:
        return image
    
def get_top_level(image: DataTree | DataArray):
    if isinstance(image, DataTree):
        return sd.get_pyramid_levels(image, n=len(image)-1)
    else:
        return image
    
def get_scaling_factor(image: DataTree | DataArray):
    base_shape = get_base_level(image).shape
    top_shape = get_top_level(image).shape
    return np.array(base_shape[1:]) / np.array(top_shape[1:])

class SpatialData(sd.SpatialData): 
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if 'images' not in self.attrs:
            self.attrs['images'] = []

    def write(self, file_path: str, **kwargs):
        sdata_copy = copy.copy(self)  # Shallow copy of the main object
        # save image name and loc to metadata then drop all images before saving
        preloaded_images = [info['name'] for info in sdata_copy.attrs['images']]
        for image_name in self.images:
            if image_name not in preloaded_images:
                sdata_copy.attrs['images'].append({'name': image_name, 
                                            'type': 'spatialdata_img',
                                            'path': sd.get_dask_backing_files(self[image_name])})
        image_names = list(sdata_copy.images.keys())
        sdata_copy.images = copy.copy(self.images)  
        for image_name in tqdm(image_names):
            sdata_copy[image_name] = Image2DModel.parse(np.zeros((1,1,1)))
        # wipe all patch arrays to 1D zero arrays before saving
        sdata_copy.tables = copy.copy(self.tables)  
        for table_name in tqdm(sdata_copy.tables):
            sdata_copy[table_name] = copy.copy(self[table_name])  # Copy table object
            sdata_copy[table_name].obsm = self[table_name].obsm.copy()  # Copy `obsm` dict

            if 'patch' in sdata_copy[table_name].obsm:
                patch_arr = sdata_copy[table_name].obsm['patch']
                sdata_copy[table_name].obsm['patch'] = da.zeros((len(patch_arr)))
        print("Writing data to", file_path)
        sd.SpatialData.write(sdata_copy, file_path, **kwargs)


def read_zarr(store, **kwargs):
    from meson import add_wsi
    sdata = sd.read_zarr(store, **kwargs)
    image_infos = copy.copy(sdata.attrs['images'])
    for image_info in image_infos:
        if isinstance(image_info['path'], list):
            image_path = image_info['path'][0]
        else:
            image_path = image_info['path']
        image_name = image_info['name']
        if image_info['type'] == 'spatialdata_image':
            sdata[image_name] = _read_multiscale(image_path, raster_type="image")
        elif image_info['type'] == 'wsi':
            add_wsi(sdata, path=image_path, image_name=image_name)

    images = {}
    for table_name, table in sdata.tables.items():
        if 'patch' not in table.obsm:
            continue
        patches = []
        for idx, row in tqdm(table.obs.iterrows()):
            image_name = row['image']
            if image_name not in images:
                image = get_base_level(sdata[row['image']])
                image = image.chunk(chunks=get_optimal_chunk_size(image)) # via xarray
                images[image_name] = image
            else:
                image = images[image_name]
            shape_element = sdata[row['region']]
            instance_id = row['instance_id']
            polygon = shape_element.loc[instance_id].geometry
            
            coords = np.array(polygon.exterior.coords)[:-1]  
            min_x, min_y = coords[0]  # Top-left corner
            size_x = int(coords[1][0] - coords[0][0])
            size_y = int(coords[2][1] - coords[0][1])
            patch = image[
                    :,  
                    int(min_y):int(min_y + size_y),
                    int(min_x):int(min_x + size_x)
            ]
            patches.append(patch)
            
        patches_array = da.stack(patches)
        table.obsm['patch'] = patches_array
        print(table)
        
    return sdata


def _add_bbox_coordinates(sdata, image_name, point_name='grid_point'):
    bbox_df = sdata[f'{image_name}_{point_name}_bbox'].copy() 
    
    coords_df = pd.DataFrame({
        'instance_id': range(len(bbox_df)),  
        'xmin': bbox_df['geometry'].apply(lambda x: int(x.exterior.coords[0][0])),
        'ymin': bbox_df['geometry'].apply(lambda x: int(x.exterior.coords[0][1])),
        'xmax': bbox_df['geometry'].apply(lambda x: int(x.exterior.coords[2][0])),
        'ymax': bbox_df['geometry'].apply(lambda x: int(x.exterior.coords[2][1]))
    })
    
    patch_obs = sdata[f'{image_name}_{point_name}_patch'].obs
    return patch_obs.merge(coords_df, on='instance_id', how='left')

def export_patch(sdata, 
                file_path: str,
                image_name: str | list = 'all'):
    all_patch_dfs = []
    if isinstance(image_name, str):
        if image_name == 'all':
            image_names = sdata.images
        else:
            image_names = [image_name]
    else:
        image_names = image_name
    for image_name in image_names:
        coords_df = _add_bbox_coordinates(sdata, image_name)
        all_patch_dfs.append(coords_df)
    combined = pd.concat(all_patch_dfs, axis=0, ignore_index=True)
    combined = combined[['image', 'xmin', 'ymin', 'xmax', 'ymax']]
    combined.to_csv(file_path)
    return combined