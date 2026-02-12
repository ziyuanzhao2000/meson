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
import pandas as pd
import joblib

from pathlib import Path
import shutil

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
    
def get_scaling_factor(image: DataTree | DataArray, level=-1):
    if level==-1:
        base_shape = get_base_level(image).shape
        top_shape = get_top_level(image).shape
        return np.array(base_shape[1:]) / np.array(top_shape[1:])
    else:
        base_shape = get_base_level(image).shape
        level_shape = sd.get_pyramid_levels(image, n=level).shape
        return np.array(base_shape[1:]) / np.array(level_shape[1:])

class SpatialData(sd.SpatialData): 
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if 'images' not in self.attrs:
            self.attrs['images'] = []
        if 'models_metadata' not in self.attrs:
            self.attrs['models_metadata'] = {}

    def write(self, file_path: str, **kwargs):
        sdata_copy = copy.copy(self)  # Shallow copy of the main object
        # save image name and loc to metadata then drop all images before saving
        if 'images' in sdata_copy.attrs:
            preloaded_images = [info['name'] for info in sdata_copy.attrs['images']]
        else:
            preloaded_images = []
            self.attrs['images'] = []
        for image_name in self.images:
            if image_name not in preloaded_images:
                sdata_copy.attrs['images'].append({'name': image_name, 
                                            'type': 'spatialdata_img',
                                            'path': sd.get_dask_backing_files(self[image_name])})
        image_names = list(sdata_copy.images.keys())
        sdata_copy.images = copy.copy(self.images)  
        for image_name in tqdm(image_names):
            sdata_copy[image_name] = Image2DModel.parse(np.zeros((1,1,1)), dims=['c', 'y', 'x'])
        # wipe all patch arrays to 1D zero arrays before saving
        sdata_copy.tables = copy.copy(self.tables)  
        for table_name in tqdm(sdata_copy.tables):
            sdata_copy[table_name] = copy.copy(self[table_name])  # Copy table object
            sdata_copy[table_name].obsm = self[table_name].obsm.copy()  # Copy `obsm` dict

            if 'patch' in sdata_copy[table_name].obsm:
                patch_arr = sdata_copy[table_name].obsm['patch']
                sdata_copy[table_name].obsm['patch'] = da.zeros((len(patch_arr)))
        # sdata uses json that cannot serialize model params
        for model_info in sdata_copy.attrs['models_metadata']:
            del sdata_copy.attrs['models'][model_info['name']]
        print("Writing data to", file_path)
        sd.SpatialData.write(sdata_copy, file_path, **kwargs)


def read_zarr(store, backend='tiffslide', **kwargs):
    from meson import add_wsi
    sdata = sd.read_zarr(store, **kwargs)
    image_infos = copy.copy(sdata.attrs['images'])
    for image_info in image_infos:
        if isinstance(image_info['path'], list) and len(image_info['path']):
            image_path = image_info['path'][0]
        else:
            image_path = image_info['path']
        image_name = image_info['name']
        if image_info['type'] == 'spatialdata_img':
            sdata[image_name] = _read_multiscale(image_path, raster_type="image")
        elif image_info['type'] == 'wsi':
            add_wsi(sdata, path=image_path, image_name=image_name, backend=backend)
    sdata.attrs['models'] = {}
    for model_info in sdata.attrs['models_metadata']:
        if model_info['serializer'] == 'joblib': 
            sdata.attrs['models'][model_info['name']] = joblib.load(model_info['path'])
    return sdata

def overwrite_element(sdata , name) -> None:
    if sdata.path is None:
        raise ValueError("sdata.path must be set (e.g., via sdata.write(path)) before overwriting.")
    tmp_name = f"{name}__tmp_overwrite"
    new_element = sdata[name]
    sdata[tmp_name] = new_element
    sdata.write_element(tmp_name)
    group_path = Path(sdata.path) / sdata.locate_element(sdata[name])[0]
    del sdata[name]
    
    if group_path.exists():
        shutil.rmtree(group_path)
    else:
        raise FileNotFoundError(f"Expected Zarr group for element '{name}' not found at {group_path}")
    
    tmp_path = Path(sdata.path) / sdata.locate_element(sdata[tmp_name])[0]
    shutil.move(str(tmp_path), str(group_path))
    sdata[name] = new_element
    del sdata[tmp_name]

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
        all_patch_dfs.append(sdata[f'{image_name}_grid_point_patch'].obs)
    combined = pd.concat(all_patch_dfs, axis=0, ignore_index=True)
    combined = combined[['image', 'xmin', 'ymin', 'xmax', 'ymax']]
    combined.to_csv(file_path)
    return combined