import openslide
import spatialdata as sd
import dask.array as da
import copy

from ._settings import settings

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

class SpatialData(sd.SpatialData): 
    def write(self, file_path: str, *args, **kwargs):
        sdata_copy = copy.copy(self)  # Shallow copy of the main object
        
        sdata_copy.tables = copy.copy(self.tables)  
        for table_name in sdata_copy.tables:
            sdata_copy[table_name] = copy.copy(self[table_name])  # Copy table object
            sdata_copy[table_name].obsm = self[table_name].obsm.copy()  # Copy `obsm` dict

            if 'patch' in sdata_copy[table_name].obsm:
                patch_arr = sdata_copy[table_name].obsm['patch']
                sdata_copy[table_name].obsm['patch'] = da.zeros((len(patch_arr)))

        sdata_copy.write(file_path, *args, **kwargs)


def from_zarr(store, *args, **kwargs):
    sdata = sd.read_zarr(store, *args, **kwargs)
    for table_name, table in sdata.tables.items():
        if 'patch' not in table.obsm:
            continue
        patches = []
        for idx, row in table.obs.iterrows():
            image = sdata[row['image']]
            
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