from pathlib import Path
from typing import Any, Literal, Tuple

import xarray
from spatialdata.transformations import set_transformation, Identity
from spatialdata import SpatialData
from spatialdata.models import Image2DModel
from spatialdata.transformations import Identity, Scale
from xarray import DataArray, Dataset, DataTree


def add_wsi(sdata: SpatialData, 
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
    chunks: tuple[int, int, int] = (3, 256, 256),
    as_image: bool = True,
    backend: Literal["tiffslide", "openslide"] = "tiffslide",
) -> SpatialData | Tuple[DataTree, str, dict]:
    """Read a WSI into a `SpatialData` object

    Args:
        path: Path to the WSI
        chunks: Tuple representing the chunksize for the dimensions `(C, Y, X)`.
        as_image: If `True`, returns a, image instead of a `SpatialData` object
        backend: The library to use as a backend in order to load the WSI. One of: `"openslide"`, `"tiffslide"`.

    Returns:
        A `SpatialData` object with a multiscale 2D-image of shape `(C, Y, X)`, or just the DataTree if `as_image=True`
    """
    if not as_image:
        import meson
        sdata = meson.SpatialData()
        add_wsi(sdata, path=path, backend=backend)
        return sdata 
    
    image_name, img, slide, slide_metadata = _open_wsi(path, backend=backend)
    # img = img.rename_dims({"S": "C", "Y": "Y", "X": "X"})
    img = img.rename_dims({"S": "C"})

    images = {}
    for level, key in enumerate(sorted(list(img.keys()), key=int)):
        suffix = key if key != "0" else ""

        scale_image = DataArray(
            img[key].transpose("C", f"Y{suffix}", f"X{suffix}"),
            dims=("c", "y", "x"),
        ).chunk(chunks)

        scale_factor = slide.level_downsamples[level]

        scale_image = Image2DModel.parse(
            scale_image[:3, :, :],
            transformations={"global": _get_scale_transformation(scale_factor)},
            c_coords=("r", "g", "b"),
        )
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
) -> SpatialData:
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
    image_model_kwargs = _default_image_models_kwargs(image_model_kwargs)

    image_name, img, _, tiff_metadata = _open_wsi(path, backend=backend)

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
    path: str | Path, backend: Literal["tiffslide", "openslide"] = "openslide"
) -> tuple[str, xarray.Dataset, Any, dict]:
    image_name = Path(path).stem

    if backend == "tiffslide":
        import tiffslide

        slide = tiffslide.open_slide(path)
        zarr_store = slide.zarr_group.store
        zarr_img = xarray.open_zarr(
            zarr_store,
            consolidated=False,
            mask_and_scale=False,
        )
    elif backend == "openslide":
        import openslide

        from ._openslide import OpenSlideStore

        slide = openslide.open_slide(path)
        zarr_store = OpenSlideStore(path).store
    else:
        raise ValueError(f"Invalid {backend:=}. Supported options are 'openslide' and 'tiffslide'")

    zarr_img = xarray.open_zarr(zarr_store, consolidated=False, mask_and_scale=False)

    metadata = {
        "properties": slide.properties,
        "dimensions": slide.dimensions,
        "level_count": slide.level_count,
        "level_dimensions": slide.level_dimensions,
        "level_downsamples": slide.level_downsamples,
    }
    return image_name, zarr_img, slide, metadata
