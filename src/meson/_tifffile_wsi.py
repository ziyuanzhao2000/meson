import zarr
import dask.array as da
import numpy as np
import uuid
import os
import shutil

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
        self.temp_dir = f".{self.uuid}"
        os.makedirs(self.temp_dir, exist_ok=True)
        self._dirty = False
        self.data_array = None
        self.mask_array = None

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
    def data(self):
        if self.dirty:
            return self
        else:
            return self.original

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
            mask_slice = self.mask_array[key]
            original_data = self.original[key]
            temp_data = self.data_array[key]
            return np.where(mask_slice, temp_data, original_data)
        else:
            return self.original[key]

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
            y_step, x_step = self.chunks[y_ax], self.chunls[x_ax]
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
                    yield self[tuple(full_index)]

class TiffPage:
    def __init__(self, tiffpage):
        self._page = tiffpage

        base_store = self._page.aszarr()
        max_bytes = 1e12 #tbd later
        cached_store = zarr.storage.LRUStoreCache(base_store, max_bytes)
        initial_array = zarr.open(cached_store)
        self._data = WriteableZarrArray(initial_array, axes=self.axes)

    def __getattr__(self, name):
        return getattr(self._page, name)

    @property
    def data(self):
        return self._data.data
    
    def _repr_html_(self):
        return self._data._repr_html_()
        
    def __repr__(self):
        return self.data.__repr__()
    
    def __str__(self):
        return self.data.__str__()

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def tiles(self, tile_size):
        return self._data.tiles(tile_size)
    
    def cleanup(self):
        if self._data is not None:
            self._data.cleanup()

class TiffLevel():
    def __init__(self, tifflevel, level_id):
        self._level = tifflevel
        self.level_id = level_id
        self._pages = [TiffPage(page) for page in tifflevel.pages]
        
        base_store = self._level.aszarr()
        max_bytes = 1e12 #tbd later
        cached_store = zarr.storage.LRUStoreCache(base_store, max_bytes)
        initial_array = zarr.open(cached_store)
        if isinstance(initial_array, zarr.hierarchy.Group): # this occurs at level 0
            initial_array = initial_array[0] 
        self._data = WriteableZarrArray(initial_array, axes=self.axes)

    @property
    def pages(self):
        return self._pages
    
    def __getattr__(self, name):
        return getattr(self._level, name)
    
    @property
    def data(self):
        return self._data.data
    
    def _repr_html_(self):
        return f"<h3>Pyramid level {self.level_id},\n</h3>" + self._data._repr_html_()

    def __repr__(self):
        return f"Pyramid level {self.level_id},\n" + self.data.__repr__()
    
    def __str__(self):
        return f"Pyramid level {self.level_id},\n" + self.data.__str__()
    
    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def tiles(self, tile_size):
        return self._data.tiles(tile_size)
    
    def cleanup(self):
        for page in self._pages:
            page.cleanup()
        if self._data is not None:
            self._data.cleanup()

class TiffSeries():
    def __init__(self, tiffseries):
        self._series = tiffseries
        self._levels = [TiffLevel(level, level_id) \
                        for level_id, level in enumerate(tiffseries.levels)]

    @property
    def levels(self):
        return self._levels
    
    def __getattr__(self, name):
        return getattr(self._series, name)

    def __repr__(self):
        s = ' \n'.join(
            s
            for s in (
                f'{self.name!r}' if self.name else '',
                'x'.join(str(i) for i in self.shape),
                str(self.dtype),
                f"Axes order: {self.axes}",
                f"Image type: {self.kind}",
                (f'{len(self.levels)} Levels') if self.is_pyramidal else '',
                f'{len(self)} Pages',
                (f'@{self.dataoffset}') if self.dataoffset else '',
            )
            if s
        )
        return f'TiffPageSeries {self._index}  {s}'
    
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

    def cleanup(self):
        for level in self._levels:
            level.cleanup()


class Slide():
    pass 
