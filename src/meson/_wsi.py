from pathlib import Path
from typing import Any, Literal, Tuple, TYPE_CHECKING

from spatialdata.transformations import set_transformation, Identity, Scale
from spatialdata.models import Image2DModel
import xarray

if TYPE_CHECKING:
    from spatialdata import SpatialData
    from xarray import Dataset, DataTree


def add_wsi(sdata: "SpatialData", 
            path: str | Path, 
            backend: Literal["tiffslide", "openslide"] = "tiffslide",
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
    backend: Literal["tiffslide", "openslide", "default"] = "tiffslide",
    type: Literal["HnE", "mIF", "mono"] = "HnE",
) -> "SpatialData" | Tuple["DataTree", str, dict]:
    """Read a WSI into a `SpatialData` object

    Args:
        path: Path to the WSI
        chunks: Tuple representing the chunksize for the dimensions `(C, Y, X)`.
        as_image: If `True`, returns a, image instead of a `SpatialData` object
        backend: The library to use as a backend in order to load the WSI. One of: `"openslide"`, `"tiffslide"`.

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
    return multiscale_image, image_name, slide_metadata



def _get_scale_transformation(scale_factor: float):
    if scale_factor == 1:
        return Identity()
    return Scale([scale_factor, scale_factor], axes=("x", "y"))


def wsi_autoscale(
    path: str | Path,
    image_model_kwargs: dict | None = None,
    backend: Literal["tiffslide", "openslide"] = "tiffslide",
) -> "SpatialData":
    """Read a WSI into a `SpatialData` object.

    Scales are generated automatically by `spatialdata` instead of using
    the default multiscales.

    Args:
        path: Path to the WSI
        image_model_kwargs: Kwargs provided to the `Image2DModel`
        backend: The library to use as a backend in order to load the WSI. One of: `"openslide"`, `"tiffslide"`.

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
    path: str | Path, backend: Literal["tiffslide", "openslide", "custom-multiseries", "default"] = "tiffslide"
) -> tuple[str, "Dataset", Any, dict]:
    image_name = Path(path).stem

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
