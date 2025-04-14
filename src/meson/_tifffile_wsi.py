import zarr
import dask.array as da
import numpy as np
import uuid
import os
import shutil
import tifffile
from fractions import Fraction
import re

tag_registries = [tifffile.TIFF.TAGS,
                  tifffile.TIFF.GPS_TAGS,
                  tifffile.TIFF.IOP_TAGS,
                  tifffile.TIFF.UIC_TAGS,
                  tifffile.TIFF.EXIF_TAGS,
                  tifffile.TIFF.NDPI_TAGS]

# tifffile_accepted_metadata = [
#     # Plane level
#     'DeltaT',
#     'DeltaTUnit',
#     'ExposureTime',
#     'ExposureTimeUnit',
#     'PositionX',
#     'PositionXUnit',
#     'PositionY',
#     'PositionYUnit',
#     'PositionZ',
#     'PositionZUnit'
#     # Channel level
#     'Name',
#     'AcquisitionMode',
#     'Color',
#     'ContrastMethod',
#     'EmissionWavelength',
#     'EmissionWavelengthUnit',
#     'ExcitationWavelength',
#     'ExcitationWavelengthUnit',
#     'Fluor',
#     'IlluminationType',
#     'NDFilter',
#     'PinholeSize',
#     'PinholeSizeUnit',
#     'PockelCellSetting',
#     # Image level
#     'Name', ?
#     'SignificantBits',
#     'PhysicalSizeX',
#     'PhysicalSizeXUnit',
#     'PhysicalSizeY',
#     'PhysicalSizeYUnit',
#     'PhysicalSizeZ',
#     'PhysicalSizeZUnit',
#     'TimeIncrement',
#     'TimeIncrementUnit',
# ]

def get_tag_name(tag_code):
    for tag_registry in tag_registries:
        if tag_code in tag_registry:
            return tag_registry[tag_code]
    return tag_code


def merge_dicts(dicts, names):
    if len(dicts) != len(names):
        raise ValueError("Number of dictionaries and names must match")
    
    if not dicts:
        return {}
    
    all_keys = set()
    for d in dicts:
        all_keys.update(d.keys())
    
    result = {}
    
    for key in all_keys:
        values = [d.get(key) for d in dicts if key in d]
        
        if len(values) == len(dicts):
            are_equal = True
            first_val = values[0]
            
            for v in values[1:]:
                if isinstance(first_val, np.ndarray) or isinstance(v, np.ndarray):
                    if not np.array_equal(first_val, v, equal_nan=True):
                        are_equal = False
                        break
                elif first_val != v:
                    are_equal = False
                    break
            
            if are_equal:
                result[key] = values[0]
                continue
        
        for i, d in enumerate(dicts):
            if key in d:
                result[f"{names[i]}{key}"] = d[key]
    
    return result

# copied from tiffslides: https://github.com/Bayer-Group/tiffslide/blob/main/tiffslide/tiffslide.py#L758
def recover_mpp(metadata):
    """recover mpp from tiff tags"""
    parsed = {}

    try:
        resolution_unit = metadata["ResolutionUnit"]
        x_resolution = Fraction(*metadata["XResolution"])
        y_resolution = Fraction(*metadata["YResolution"])
    except KeyError:
        pass
    else:
        RESUNIT = tifffile.RESUNIT
        scale = {
            RESUNIT.INCH: 25400.0,
            RESUNIT.CENTIMETER: 10000.0,
            RESUNIT.MILLIMETER: 1000.0,
            RESUNIT.MICROMETER: 1.0,
            RESUNIT.NONE: None,
        }.get(resolution_unit, None)
        if scale is not None:
            try:
                mpp_x = scale / x_resolution
                mpp_y = scale / y_resolution
            except ArithmeticError:
                pass
            else:
                parsed['PhysicalSizeX'] = mpp_x
                parsed['PhysicalSizeY'] = mpp_y
                parsed['PhysicalSizeXUnit'] = 'µm'
                parsed['PhysicalSizeYUnit'] = 'µm'
    return parsed

def TIFFParser(metadata):
    return recover_mpp(metadata)

def NDPIParser(metadata):
    return metadata

class WriteableZarrArray:
    def __init__(self, original_array, axes=None):
        self.original = original_array
        self.axes = axes
        if axes is None:
            num_axes = len(original_array.shape)
            if num_axes==2:
                self.axes = 'YX'
            elif num_axes>2:
                self.axes = '?'*(num_axes-2)+'YX'
        if isinstance(original_array, np.ndarray):
            self.original_da = da.from_array(original_array)
        elif isinstance(original_array, zarr.core.Array) or \
            isinstance(original_array, zarr.storage.LRUStoreCache):
            self.original_da = da.from_zarr(original_array)
        self.uuid = str(uuid.uuid4())
        self.temp_dir = f".temp_{self.uuid}"
        os.makedirs(self.temp_dir, exist_ok=True)
        self._dirty = False

    def _create_temp_arrays(self):
        # Create temp arrays on disk with same properties as original
        # might want to customize compression later if needed
        if not hasattr(self, 'data_array'):
            self.data_array = zarr.open(os.path.join(self.temp_dir, 'data'), 
                                    shape=self.original.shape,
                                    chunks=self.chunks,
                                    dtype=self.original.dtype,
                                    mode='w')
            
            self.mask_array = zarr.open(os.path.join(self.temp_dir, 'mask'), 
                                    shape=self.original.shape,
                                    chunks=self.chunks,
                                    dtype=bool,
                                    mode='w')

    @property
    def dirty(self):
        return self._dirty

    @dirty.setter
    def dirty(self, value):
        if value and not self._dirty:
            self._create_temp_arrays()
        self._dirty = value

    def _repr_html_(self):
        if hasattr(self.original, "_repr_html_"):
            return self.original._repr_html_()
        elif hasattr(self, 'original_da'):
            return self.original_da._repr_html_()
        else:
            return self.__repr__()
        
    def __repr__(self):
        return self.original.__repr__()
    
    def __str__(self):
        return self.original.__str__()

    def __getitem__(self, key):
        if self.dirty:
            mask_slice = da.array(self.mask_array)[key]
            original_data = da.array(self.original)[key]
            temp_data = da.array(self.data_array)[key]
            return da.where(mask_slice, temp_data, original_data)
        else:
            return da.array(self.original)[key]

    def __setitem__(self, key, value):
        self.dirty = True
        self.data_array[key] = value
        self.mask_array[key] = True

    @property
    def shape(self):
        return self.original.shape

    @property
    def chunks(self):
        original = self.original
        if isinstance(original, zarr.core.Array):
            return original.chunks
        elif isinstance(original, da.core.Array):
            return original.chunksize

    @property
    def dtype(self):
        return self.original.dtype

    def cleanup(self):
        shutil.rmtree(self.temp_dir)

    def tiles(self, tile_size=None):
        y_ax, x_ax = self.axes.index('Y'), self.axes.index('X')
        y_size, x_size = self.original.shape[y_ax], self.original.shape[x_ax]
        if tile_size is None:
            y_step, x_step = self.chunks[y_ax], self.chunks[x_ax]
        else:
            y_step, x_step = tile_size
        shape = self.shape
        iter_axes = [i for i in range(len(shape)) if i not in (y_ax, x_ax)]
        iter_shape = [shape[i] for i in iter_axes]
        for index in np.ndindex(*iter_shape):
            for y in range(0, y_size, y_step):
                for x in range(0, x_size, x_step):
                    full_index = list(index)
                    if y_ax > x_ax:
                        full_index.insert(x_ax, slice(x, x+x_step, 1))
                        full_index.insert(y_ax, slice(y, y+y_step, 1))
                    else:
                        full_index.insert(y_ax, slice(y, y+y_step, 1))
                        full_index.insert(x_ax, slice(x, x+x_step, 1))
                    if self.dirty:
                        yield self[tuple(full_index)].compute()
                    else:
                        yield self.original[tuple(full_index)]

class TiffPage:
    def __init__(self, tiffpage):
        self._page = tiffpage

        base_store = self._page.aszarr()
        max_bytes = 1e8 #tbd later
        cached_store = zarr.storage.LRUStoreCache(base_store, max_bytes)
        initial_array = zarr.open(cached_store)
        self.data = WriteableZarrArray(initial_array, axes=self.axes)

    def __getattr__(self, name):
        return getattr(self._page, name)


    def _repr_html_(self):
        return self.data._repr_html_()
        
    def __repr__(self):
        return self.data.__repr__()
    
    def __str__(self):
        return self.data.__str__()

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def tiles(self, tile_size):
        return self.data.tiles(tile_size)
    
    def cleanup(self):
        if self.data is not None:
            self.data.cleanup()

class TiffLevel():
    def __init__(self, tifflevel, level_id):
        self._level = tifflevel
        self.level_id = level_id
        self._pages = [TiffPage(page) for page in tifflevel.pages]
        
        base_store = self._level.aszarr()
        max_bytes = 1e8 #tbd later
        cached_store = zarr.storage.LRUStoreCache(base_store, max_bytes)
        initial_array = zarr.open(cached_store)
        if isinstance(initial_array, zarr.hierarchy.Group): # this occurs at level 0
            initial_array = initial_array[0] 
        self.data = WriteableZarrArray(initial_array, axes=self.axes)

        self._metadata = dict([
            (get_tag_name(tag.code), tag.value) for tag in self._level.pages[0].tags
        ])
        self._parsed_metadata = {}

    @property
    def metadata(self):
        metadata = {}
        metadata.update(self._parsed_metadata)
        metadata.update(self._metadata)
        return metadata

    @property
    def pages(self):
        return self._pages
    
    def __getattr__(self, name):
        if name in ['name']:
            attr = self.metadata.get(name)
            if attr is not None:
                return attr
        return getattr(self._level, name)

    @property
    def is_multiscale(self):
        return (self.is_pyramidal or self.level_id > 0)
    
    def _repr_html_(self):
        if self.is_multiscale:
            return f"<h3>Pyramid level {self.level_id},\n</h3>" + self.data._repr_html_()
        else:
            return self.data._repr_html_()

    def __repr__(self):
        return f"Pyramid level {self.level_id},\n" + self.data.__repr__()
    
    def __str__(self):
        return f"Pyramid level {self.level_id},\n" + self.data.__str__()
    
    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def tiles(self, tile_size):
        return self.data.tiles(tile_size)
    
    def parse_metadata(self, kind):
        metadata = {}
        metadata.update(TIFFParser(self._metadata))
        if kind == 'ndpi':
            parser = NDPIParser
        else:
            parser = lambda x: x
        self._parsed_metadata = parser(metadata)

    def cleanup(self):
        for page in self._pages:
            page.cleanup()
        if self.data is not None:
            self.data.cleanup()

class TiffSeries():
    def __init__(self, tiffseries):
        self._series = tiffseries
        self._levels = [TiffLevel(level, level_id) \
                        for level_id, level in enumerate(tiffseries.levels)]

    @property
    def metadata(self):
        if len(self._levels) == 1:
            return self._levels[0].metadata
        else:
            return merge_dicts(dicts=[level.metadata for level in self._levels],
                               names=[f'level_{i}.' for i in range(len(self._levels))])

    @property
    def levels(self):
        return self._levels
    
    @property
    def name(self):
        if hasattr(self._series, 'name'):
            return self._series.name
        else:
            return self._levels[0].name
    
    def __getattr__(self, name):
        return getattr(self._series, name)

    # alias
    @property
    def is_multiscale(self):
        return self.is_pyramidal
    

    def __repr__(self):
        lines = [
                f'Image {self.name!r}' if self.name else 'Image' + f'of type {self.kind}',
                f'Data type: {str(self.dtype)}',
                f"Axes order: {self.axes}",
                f'Pyramidal with {len(self.levels)} levels:' if self.is_multiscale else '',
            ]
        if self.is_multiscale:
            for level_id, level in enumerate(self.levels):
                lines.append(f'  Level {level_id}, data shape: {level.shape}, chunk shape: {level.data.chunks}')
        else:
            lines.append(f'Data shape: {self.levels[0].shape}')

        return ' \n'.join(s for s in lines if s)
    
    def __getitem__(self, key):
        if isinstance(key, int) or len(key) == 0:
            return self.levels[key]
        else:
            return self.levels[key[0]][key[1:]]

    def __setitem__(self, key, value):
        if isinstance(key, int) or len(key) == 0:
            self.levels[key] = value
        else:
            self.levels[key[0]][key[1:]] = value

    def parse_metadata(self, kind):
        for level in self._levels:
            level.parse_metadata(kind)

    def cleanup(self):
        for level in self._levels:
            level.cleanup()


class TiffFile():
    def __init__(self, file, kind=None, *args, **kwargs):
        self._file = file
        self._tifffile = tifffile.TiffFile(file, *args, **kwargs)
        self._series = [TiffSeries(series) for series in self._tifffile.series]
        self._kind = kind if kind else self.series[0].kind 

        for series in self._series:
            series.parse_metadata(self._kind)

    @property
    def series(self):
        return self._series

    @property
    def kind(self):
        return self._kind 

    def __getattr__(self, name):
        return getattr(self._tifffile, name)

    def __repr__(self):
        lines = [f'TiffFile ({self.kind}) from {self._file} with {len(self.series)} image series: ']
        for series in self._series:
            lines.append(f'  Series {series.name!r} with {len(series.levels)} levels')
        return ' \n'.join(s for s in lines if s)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._series[key]
        elif isinstance(key, str):
            for series in self.series:
                if series.name == key:
                    return series

    def __setitem__(self, key, value):
        if isinstance(key, int):
            self._series[key] = value
        elif isinstance(key, str):
            for id, series in enumerate(self.series):
                if series.name == key:
                    self._series[id] = value
                    break

    def close(self):
        for series in self.series:
            series.cleanup()
        self._tifffile.close()

def remove_invalid_xml_chars(text):
    xml_compliant_text = re.sub(r'[^\x09\x0A\x0D\x20-\uD7FF\uE000-\uFFFD\U00010000-\U0010FFFF]+', '', text)
    return xml_compliant_text

class TiffWriter(tifffile.TiffWriter):
    def __init__(self, file, *args, **kwargs):
        self.file = file
        super().__init__(file, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._tifffile, name)
    
    def write(self, 
            tiff_obj, 
            metadata={},
            write_tiles=True, 
            *args, **kwargs):
        
        if isinstance(tiff_obj, TiffLevel):
            level = tiff_obj
            serialized = dict([(remove_invalid_xml_chars(str(key)), 
                                remove_invalid_xml_chars(str(val))) for key, val in level.metadata.items()])
            metadata.update(serialized)
            metadata.update({'MapAnnotation': serialized,})
            tile_size = kwargs.get("tile", (1024, 1024))
            kwargs['tile']=tile_size
            shape = level.data.shape
            if write_tiles:
                data = level.data.tiles(tile_size)  
                if level.axes == 'YXS':
                    shape = tuple([shape[i] for i  in [2, 0, 1]])
            else:
                if level.data.dirty:
                    data = level.data.compute()
                else:
                    data = level.data.original[:]
            if level.level_id == 0: # base level
                metadata.update({'Name': level.name})
            super().write(data, 
                        metadata=metadata, 
                        shape=shape,
                        dtype=level.data.dtype,
                        *args, **kwargs)
        elif isinstance(tiff_obj, TiffSeries):
            series = tiff_obj
            subresolutions = len(series.levels) - 1
            if subresolutions == 0:
                self.write(series.levels[0], 
                        metadata=metadata, 
                        write_tiles=write_tiles,
                        *args,
                        **kwargs)
            else:
                self.write(series.levels[0], 
                        metadata=metadata, 
                        write_tiles=write_tiles,
                        subifds=subresolutions,
                        *args,
                        **kwargs)
                for level in series.levels[1:]:
                    self.write(level, 
                        metadata=metadata, 
                        write_tiles=write_tiles,
                        subfiletype=1,
                        *args,
                        **kwargs)
        elif isinstance(tiff_obj, TiffFile):
            file = tiff_obj
            for series in file.series:
                self.write(series, 
                           metadata=metadata,
                           write_tiles=write_tiles,
                           *args,
                           **kwargs)