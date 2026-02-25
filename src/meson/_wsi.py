from pathlib import Path
from typing import Any, Literal, Tuple, TYPE_CHECKING

from spatialdata.transformations import set_transformation, Identity, Scale
from spatialdata.models import Image2DModel
import xarray

if TYPE_CHECKING:
    from spatialdata import SpatialData
    from xarray import Dataset, DataTree


def _tifffile_wza_to_cyx(wza) -> "xarray.DataArray | dask.array.Array | np.ndarray":
    """Convert a WriteableZarrArray to a (C, Y, X) array.

    Calls wza.to_array() which returns either:
    - a numpy array  (if wza.load() has been called — hot cache, zero overhead)
    - a dask array   (otherwise — fully lazy)

    xarray.DataArray accepts both without any isinstance branching, so the
    caller can always just do:
        DataArray(_tifffile_wza_to_cyx(wza), dims=['C', dim_y, dim_x])

    The transpose is done with np.transpose or da.transpose depending on the
    type, via a small unified helper so neither path loads data eagerly.
    """
    import numpy as np
    import dask.array as da

    arr = wza.to_array()   # numpy or dask — decided by loaded state
    axes = wza.axes

    def _transpose(a, order):
        if isinstance(a, np.ndarray):
            return np.transpose(a, order)
        else:
            return da.transpose(a, order)

    def _expand(a):
        if isinstance(a, np.ndarray):
            return np.expand_dims(a, axis=0)
        else:
            return da.expand_dims(a, axis=0)

    if axes == 'YXS':
        return _transpose(arr, (2, 0, 1))   # (Y,X,S) → (S,Y,X) i.e. (C,Y,X)
    elif axes == 'YX':
        return _expand(arr)                  # (Y,X) → (1,Y,X)
    elif axes[0] in ('S', 'C'):
        return arr                           # already (C,Y,X)
    else:
        # Generic: move Y and X to last two positions, flatten rest into C
        y_idx = axes.index('Y')
        x_idx = axes.index('X')
        other = [j for j in range(len(axes)) if j not in (y_idx, x_idx)]
        arr = _transpose(arr, other + [y_idx, x_idx])
        n_c = int(np.prod([arr.shape[k] for k in range(len(other))])) if other else 1
        if isinstance(arr, np.ndarray):
            return arr.reshape(n_c, arr.shape[-2], arr.shape[-1])
        else:
            return arr.reshape(n_c, arr.shape[-2], arr.shape[-1])


def add_wsi(sdata: "SpatialData", 
            path: str | Path, 
            backend: Literal["tiffslide", "openslide", "tifffile"] = "tiffslide",
            image_name: str | None = None, 
            *args, **kwargs):
    wsi_img, slide_name, slide_metadata = read_wsi(path=path, backend=backend, *args, **kwargs)
    if image_name is None:
        image_name = slide_name
     # design choice: each image gets their own coordinate system besides the global one
    set_transformation(wsi_img, Identity(), image_name) 
    sdata[image_name] = wsi_img
    sdata[image_name].attrs["metadata"] = slide_metadata
    sdata[image_name].attrs["backend"] = backend
    sdata[image_name].name = image_name
    sdata.attrs['images'].append({'name': image_name, 'type': 'wsi', 'path': path})

def read_wsi(
    path: str | Path,
    chunks: tuple[int, int, int] | dict = {},
    as_image: bool = True,
    backend: Literal["tiffslide", "openslide", "tifffile", "default"] = "tiffslide",
    type: Literal["HnE", "mIF", "mono"] = "HnE",
) -> "SpatialData" | Tuple["DataTree", str, dict]:
    """Read a WSI into a `SpatialData` object

    Args:
        path: Path to the WSI
        chunks: Tuple representing the chunksize for the dimensions `(C, Y, X)`.
        as_image: If `True`, returns a, image instead of a `SpatialData` object
        backend: The library to use as a backend in order to load the WSI. One of: `"openslide"`, `"tiffslide"`, `"tifffile"`.

    Returns:
        A `SpatialData` object with a multiscale 2D-image of shape `(C, Y, X)`, or just the DataTree if `as_image=True`
    """
    from xarray import DataArray, Dataset, DataTree
    if not as_image:
        import meson
        sdata = meson.SpatialData()
        add_wsi(sdata, path=path, backend=backend)
        return sdata 
    
    image_name, img, slide_metadata = _open_wsi(path, backend=backend)
    # img = img.rename_dims({"S": "C", "Y": "Y", "X": "X"})
    try:
        img = img.rename_dims({"S": "C"})
    except:
        pass

    images = {}
    for level, key in enumerate(sorted(list(img.keys()), key=int)):
        suffix = key if int(key) != 0 else ""
        if type == "HnE" or type == "mIF":
            scale_image = DataArray(
                img[key].transpose("C", f"Y{suffix}", f"X{suffix}"),
                dims=("c", "y", "x"),
            ).chunk(chunks)
        elif type == "mono":
            scale_image = DataArray(
                img[key].expand_dims("C", axis=0).transpose("C", f"Y{suffix}", f"X{suffix}"),
                dims=("c", "y", "x"),
            ).chunk(chunks)
        else:
            raise ValueError(f"Invalid {type:=}. Supported options are 'HnE', 'mIF', and 'mono'")

        scale_factor = slide_metadata['level_downsamples'][level]

        if type == "HnE":
            scale_image = Image2DModel.parse(
                scale_image[:3, :, :],
                transformations={"global": _get_scale_transformation(scale_factor)},
                c_coords=("r", "g", "b"),
            )
        elif type == "mIF":
            scale_image = Image2DModel.parse(
                scale_image,
                transformations={"global": _get_scale_transformation(scale_factor)},
                c_coords=tuple(f"c{i}" for i in range(scale_image.shape[0])),
            )
        elif type == "mono":
            scale_image = Image2DModel.parse(
                scale_image,
                transformations={"global": _get_scale_transformation(scale_factor)},
                c_coords=("c"),
            )
        else:
            raise ValueError(f"Invalid {type:=}. Supported options are 'HnE', 'mIF', and 'mono'")
        scale_image.coords["y"] = scale_factor * scale_image.coords["y"]
        scale_image.coords["x"] = scale_factor * scale_image.coords["x"]

        images[f"scale{key}"] = Dataset({"image": scale_image})

    multiscale_image = DataTree.from_dict(images)
    if hasattr(img, 'attrs'):
        multiscale_image.attrs = img.attrs
    return multiscale_image, image_name, slide_metadata



def preload_wsi(
    img: "DataTree | SpatialData",
    image_name: str | None = None,
    levels: list[int] | None = None,
) -> "DataTree | SpatialData":
    """Force-load WSI pyramid levels into RAM for fast repeated access.

    After calling this, the DataArrays in the DataTree are backed by numpy
    arrays rather than dask/zarr graphs, so any indexing is direct memory
    access with no scheduling overhead.

    Mutates in-place and returns the same object for optional chaining:
        wsi = preload_wsi(read_wsi(path, backend='tifffile'), levels=[2, 3])

    Only works for DataTrees loaded with backend='tifffile'.

    Args:
        img:         DataTree (from read_wsi) or SpatialData object.
        image_name:  Required when img is a SpatialData object.
        levels:      Pyramid levels to load, e.g. [2, 3, 4].
                     Defaults to all levels. Level 0 is full resolution —
                     be careful with RAM on large slides.
    """
    from xarray import DataArray, Dataset

    # Unwrap SpatialData → DataTree
    if hasattr(img, 'images'):
        if image_name is None:
            raise ValueError("image_name must be provided when passing a SpatialData object")
        datatree = img.images[image_name]
    else:
        datatree = img

    level_data_refs = datatree.attrs.get('_wsi_level_data')
    if level_data_refs is None:
        raise ValueError(
            "No _wsi_level_data found on this DataTree. "
            "preload_wsi only works with backend='tifffile'."
        )

    levels_to_load = [str(l) for l in levels] if levels is not None else list(level_data_refs.keys())

    for level_key in levels_to_load:
        if level_key not in level_data_refs:
            raise ValueError(f"Level {level_key!r} not found. Available: {list(level_data_refs.keys())}")

        wza = level_data_refs[level_key]
        wza.load()  # compute once → stores in wza._numpy

        # wza.to_array() now returns wza._numpy directly (zero-copy view).
        # _tifffile_wza_to_cyx applies np.transpose/expand_dims (not da.*),
        # so the result is a plain numpy array — xarray indexes it at RAM speed.
        scale_key = f"scale{level_key}"
        old_image = datatree[scale_key]['image']

        new_image = DataArray(
            _tifffile_wza_to_cyx(wza),       # numpy array after load()
            dims=old_image.dims,              # preserve (c, y, x)
            coords=old_image.coords,          # preserve spatial coords
            attrs=old_image.attrs,            # preserve spatialdata transforms
        )
        datatree[scale_key].ds = Dataset({"image": new_image})

    return img


def _get_scale_transformation(scale_factor: float):
    if scale_factor == 1:
        return Identity()
    return Scale([scale_factor, scale_factor], axes=("x", "y"))


def wsi_autoscale(
    path: str | Path,
    image_model_kwargs: dict | None = None,
    backend: Literal["tiffslide", "openslide", "tifffile"] = "tiffslide",
) -> "SpatialData":
    """Read a WSI into a `SpatialData` object.

    Scales are generated automatically by `spatialdata` instead of using
    the default multiscales.

    Args:
        path: Path to the WSI
        image_model_kwargs: Kwargs provided to the `Image2DModel`
        backend: The library to use as a backend in order to load the WSI. One of: `"openslide"`, `"tiffslide"`, `"tifffile"`.

    Returns:
        A `SpatialData` object with a 2D-image of shape `(C, Y, X)`
    """
    from spatialdata import SpatialData
    image_model_kwargs = _default_image_models_kwargs(image_model_kwargs)

    image_name, img, tiff_metadata = _open_wsi(path, backend=backend)

    img = img.rename_dims({"S": "c", "Y": "y", "X": "x"})

    multiscale_image = Image2DModel.parse(
        img["0"].transpose("c", "y", "x"),
        transformations={"global": Identity()},
        c_coords=("r", "g", "b"),
        **image_model_kwargs,
    )
    multiscale_image.attrs["metadata"] = tiff_metadata
    multiscale_image.attrs["backend"] = backend

    return SpatialData(images={image_name: multiscale_image})
                       #attrs={SopaAttrs.TISSUE_SEGMENTATION: image_name})


def _default_image_models_kwargs(image_models_kwargs: dict | None) -> dict:
    image_models_kwargs = {} if image_models_kwargs is None else image_models_kwargs

    if "chunks" not in image_models_kwargs:
        image_models_kwargs["chunks"] = (3, 4096, 4096)

    if "scale_factors" not in image_models_kwargs:
        image_models_kwargs["scale_factors"] = [2, 2, 2, 2]

    return image_models_kwargs


def _open_wsi(
    path: str | Path, backend: Literal["tiffslide", "openslide", "tifffile", "custom-multiseries", "default"] = "tiffslide"
) -> tuple[str, "Dataset", Any, dict]:
    image_name = Path(path).stem

    if backend == "tifffile":
        from meson._tifffile_wsi import TiffFile
        import numpy as np

        wsi = TiffFile(path)
        img_ds = {}
        level_data_refs = {}  # keep WriteableZarrArray refs alive

        series = wsi.series[0]

        metadata = {
            "properties": series.metadata,
            "dimensions": (series.levels[0].width, series.levels[0].height),
            "level_count": len(series.levels),
            "level_dimensions": [(level.width, level.height) for level in series.levels],
            "level_downsamples": [],
        }

        base_width, base_height = series.levels[0].width, series.levels[0].height
        for level in series.levels:
            downsample_x = base_width / level.width
            downsample_y = base_height / level.height
            metadata["level_downsamples"].append(float(np.round((downsample_x + downsample_y) / 2)))

        for i, level in enumerate(series.levels):
            dim_y = f'Y{i}' if i > 0 else 'Y'
            dim_x = f'X{i}' if i > 0 else 'X'

            wza = level.data
            level_data_refs[str(i)] = wza

            # _tifffile_wza_to_cyx returns numpy (if loaded) or dask (if lazy).
            # xarray.DataArray accepts both natively — no branching needed here.
            img_ds[str(i)] = xarray.DataArray(
                _tifffile_wza_to_cyx(wza),
                dims=['C', dim_y, dim_x],
            )

        zarr_img = xarray.Dataset(img_ds)
        zarr_img.attrs['_wsi_handle'] = wsi
        zarr_img.attrs['_wsi_level_data'] = level_data_refs

        return image_name, zarr_img, metadata

    if backend == "custom-multiseries":
        from meson._tifffile_wsi import TiffFile
        import numpy as np
        wsi = TiffFile(path)
        img_ds = {}
        metadata = {}
        y0, x0 = wsi[0].shape[1:]
        downsamples = []
        for i in range(len(wsi.series)):
            dim_y = f'Y{i}' if i > 0 else 'Y'
            dim_x = f'X{i}' if i > 0 else 'X'
            y, x = wsi[i].shape[1:]
            #print(y0, x0, y, x, type(y0), type(x0), type(y), type(x))
            scale_y = np.round(y0 / y)
            scale_x = np.round(x0 / x)
            assert scale_y == scale_x, "Scale factors must be equal in both dimensions"
            downsamples.append(scale_y)
            img_arr = xarray.DataArray(wsi[i][0].data[:], 
                                    dims=['C', dim_y, dim_x])
            img_ds[i] = img_arr
            metadata[f'scale{i}'] = wsi[i].metadata
        metadata['level_downsamples'] = downsamples
        zarr_img = xarray.Dataset(img_ds)
        wsi.close()
    else:
        if backend == "tiffslide":
            import tiffslide
            slide = tiffslide.open_slide(path)
            zarr_store = slide.zarr_group.store
        elif backend == "openslide":
            import openslide
            from ._openslide import OpenSlideStore
            slide = openslide.open_slide(path)
            zarr_store = OpenSlideStore(path).store
        elif backend == "custom-multiseries":
            from meson._tifffile_wsi import TiffFile
            wsi = TiffFile(path)

            wsi.close()
        elif backend == "default":
            zarr_store = None
        else:
            raise ValueError(f"Invalid {backend:=}. Supported options are 'openslide' and 'tiffslide'")
        if slide is not None:
            metadata = {
                "properties": slide.properties,
                "dimensions": slide.dimensions,
                "level_count": slide.level_count,
                "level_dimensions": slide.level_dimensions,
                "level_downsamples": slide.level_downsamples,
            }
        zarr_img = xarray.open_zarr(zarr_store, consolidated=False, mask_and_scale=False)

        
    return image_name, zarr_img, metadata
