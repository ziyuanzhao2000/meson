### Extended from Jeremy Muhlich's original code 


from __future__ import print_function, division
import warnings
import sys
import os
import re
import io
import argparse
import pathlib
import struct
import itertools
import uuid
import multiprocessing
import concurrent.futures
import numpy as np
import tifffile
import zarr
import skimage.transform

try:
    from skimage.util.dtype import _convert as dtype_convert
except ImportError:
    from skimage.util.dtype import convert as dtype_convert


def format_shape(shape):
    return "%d x %d" % (shape[1], shape[0])


def error(path, msg):
    print(f"\nERROR: {path}: {msg}")
    sys.exit(1)


class Uint16ToUint8Wrapper:
    """Wraps a zarr/array-like and converts uint16 -> uint8 on read."""

    def __init__(self, img):
        self.img = img

    def __getitem__(self, key):
        tile = self.img[key]
        return (tile >> 8).astype(np.uint8)


class SampleSplitter:

    def __init__(self, zimg, channel):
        self.zimg = zimg
        self.channel = channel

    def __getitem__(self, key):
        assert isinstance(key, tuple) and len(key) == 2, "Must index with 2-tuple"
        return self.zimg[key + (self.channel,)]


def assemble_pyramid(
    img_arrays,
    out_path,
    pixel_size=None,
    channel_names=None,
    tile_size=1024,
    split_rgb=False,
    is_mask=False,
    num_threads=0,
    verbose=True,
):
    """
    Assemble a pyramidal OME-TIFF from one or more numpy arrays or array-likes.

    Parameters
    ----------
    img_arrays : np.ndarray or list of np.ndarray
        Input image array(s). Each array can have shape:
            - (H, W)         -> single grayscale channel
            - (H, W, 3)      -> RGB image
            - (C, H, W)      -> multi-channel image
        All arrays must have the same (H, W) and dtype.
    out_path : str or pathlib.Path
        Output .ome.tif file path. Must not already exist.
    pixel_size : float, optional
        Pixel size in microns for OME-XML metadata.
    channel_names : list of str, optional
        Channel names for OME-XML metadata.
    tile_size : int, optional
        Tile width in pixels (must be a multiple of 16). Default is 1024.
    split_rgb : bool, optional
        Split RGB images into 3 discrete channels. Default is False.
    is_mask : bool, optional
        Use nearest-neighbor downsampling (for label/binary masks). Default is False.
    num_threads : int, optional
        Number of worker threads. Default 0 = auto-detect from CPU count.
    verbose : bool, optional
        Print progress information. Default is True.
    """
    out_path = pathlib.Path(out_path)
    if out_path.exists():
        error(out_path, "Output file already exists, remove before continuing.")

    if num_threads == 0:
        if hasattr(os, 'sched_getaffinity'):
            num_threads = len(os.sched_getaffinity(0))
        else:
            num_threads = multiprocessing.cpu_count()
        if verbose:
            print(f"Using {num_threads} worker threads based on detected CPU count.\n")

    tifffile.TIFF.MAXWORKERS = num_threads
    tifffile.TIFF.MAXIOWORKERS = num_threads * 5

    # Normalize input to a list of arrays
    if isinstance(img_arrays, np.ndarray):
        img_arrays = [img_arrays]

    in_imgs = []
    num_channels = 0
    base_shape = None
    base_dtype = None
    base_rgb = None

    if verbose:
        print("Scanning input images")

    for i, arr in enumerate(img_arrays, 1):
        if arr.ndim == 2:
            # (H, W) -> single grayscale
            shape = arr.shape
            dtype = arr.dtype
            is_rgb = False
            imgs = [arr]
            channels = 1
        elif arr.ndim == 3 and arr.shape[2] == 3:
            # (H, W, 3) -> RGB
            shape = arr.shape[:2]
            dtype = arr.dtype
            is_rgb = True
            channels = 3 if split_rgb else 1
            if split_rgb:
                imgs = [arr[:, :, c] for c in range(3)]
            else:
                imgs = [arr]
        elif arr.ndim == 3:
            # (C, H, W) -> multi-channel
            shape = arr.shape[1:]
            dtype = arr.dtype
            is_rgb = False
            channels = arr.shape[0]
            imgs = [arr[c] for c in range(channels)]
        else:
            raise ValueError(
                f"Array {i} has unsupported shape {arr.shape}. "
                "Expected (H,W), (H,W,3), or (C,H,W)."
            )

        if i == 1:
            base_shape = shape
            base_dtype = dtype
            base_rgb = is_rgb and not split_rgb
            if dtype in (np.uint32, np.int32):
                if not is_mask:
                    error(
                        f"array {i}",
                        "32-bit images are only supported in mask mode."
                    )
            elif dtype not in (np.uint16, np.uint8):
                error(f"array {i}", f"Can't handle dtype '{dtype}' yet.")
        else:
            if shape != base_shape:
                error(
                    f"array {i}",
                    f"Expected shape {base_shape}, got {shape}."
                )
            if dtype != base_dtype:
                if base_dtype == np.uint8 and dtype == np.uint16:
                    if verbose:
                        print(
                            f"WARNING: array {i}: dtype '{dtype}' does not match"
                            f" first input dtype '{base_dtype}'."
                            " Converting uint16 to uint8 by dividing by 257."
                        )
                    imgs = [Uint16ToUint8Wrapper(img) for img in imgs]
                else:
                    error(
                        f"array {i}",
                        f"Expected dtype '{base_dtype}', got '{dtype}'."
                    )

        if verbose:
            f_channels = 'RGB' if (is_rgb and not split_rgb) else channels
            if is_rgb and split_rgb:
                f_channels = '3 (RGB-split)'
            print(f"    array {i}: shape={shape} dtype={dtype}, channels={f_channels}")

        in_imgs.extend(imgs)
        num_channels += channels

    if verbose:
        print()

    num_levels = int(max(np.ceil(np.log2(max(base_shape) / tile_size)) + 1, 1))
    factors = 2 ** np.arange(num_levels)
    shapes = np.ceil(np.array(base_shape) / factors[:, None]).astype(int)
    cshapes = np.ceil(shapes / tile_size).astype(int)

    if channel_names and len(channel_names) != num_channels:
        error(
            out_path,
            f"Number of channel names ({len(channel_names)}) does not"
            f" match number of channels ({num_channels})."
        )

    if verbose:
        print("Pyramid level sizes:")
        for i, shape in enumerate(shapes):
            print(f"    level {i + 1}: {format_shape(shape)}", end="")
            if i == 0:
                print(" (original size)", end="")
            print()
        print()

    pool = concurrent.futures.ThreadPoolExecutor(num_threads)

    def tiles0():
        ts = tile_size
        ch, cw = cshapes[0]
        for c, img in enumerate(in_imgs, 1):
            if verbose:
                print(f"    channel {c}")
            for j in range(ch):
                for i_idx in range(cw):
                    tile = img[ts * j: ts * (j + 1), ts * i_idx: ts * (i_idx + 1)]
                    # Ensure numpy array
                    if not isinstance(tile, np.ndarray):
                        tile = np.array(tile)
                    yield tile

    def tiles(level):
        tiff_out = tifffile.TiffFile(out_path, is_ome=False)
        series = tiff_out.series[0]
        zimg = zarr.open(series.aszarr(level=level - 1))
        ts = tile_size * 2

        def tile(coords):
            c, j, i_idx = coords
            if series.axes in ("YX", "YXS"):
                assert c == 0
                t = zimg[ts * j: ts * (j + 1), ts * i_idx: ts * (i_idx + 1)]
            else:
                t = zimg[c, ts * j: ts * (j + 1), ts * i_idx: ts * (i_idx + 1)]
            if is_mask:
                t = t[::2, ::2]
            else:
                f = (2, 2)
                if base_rgb:
                    f += (1,)
                t = skimage.transform.downscale_local_mean(t, f)
                t = np.round(t).astype(base_dtype)
            return t

        ch, cw = cshapes[level]
        coords = itertools.product(range(num_channels), range(ch), range(cw))
        yield from pool.map(tile, coords)

    metadata = {"UUID": uuid.uuid4().urn}
    if pixel_size:
        metadata.update({
            "PhysicalSizeX": pixel_size,
            "PhysicalSizeXUnit": "µm",
            "PhysicalSizeY": pixel_size,
            "PhysicalSizeYUnit": "µm",
        })
    if channel_names:
        channel_list = []
        for name in channel_names:
            if name.startswith("HE"):
                channel_list.append({"Name": name, "IlluminationType": "Transmitted"})
            else:
                channel_list.append({"Name": name, "IlluminationType": "Epifluorescence"})
        metadata["Channel"] = channel_list

    photometric = "rgb" if base_rgb else "minisblack"

    if verbose:
        print(f"Writing level 1: {format_shape(shapes[0])}")

    with tifffile.TiffWriter(out_path, ome=True, bigtiff=True) as writer:
        wshape = (num_channels,) + tuple(shapes[0])
        if base_rgb:
            wshape += (3,)
        writer.write(
            data=tiles0(),
            shape=wshape,
            subifds=num_levels - 1,
            dtype=base_dtype,
            photometric=photometric,
            tile=(tile_size, tile_size),
            compression="adobe_deflate",
            predictor=True,
            metadata=metadata,
        )
        if verbose:
            print()
        for level, shape in enumerate(shapes[1:], 1):
            if verbose:
                print(f"Resizing image for level {level + 1}: {format_shape(shape)}")
            wshape = (num_channels,) + tuple(shape)
            if base_rgb:
                wshape += (3,)
            writer.write(
                data=tiles(level),
                shape=wshape,
                subfiletype=1,
                dtype=base_dtype,
                photometric=photometric,
                tile=(tile_size, tile_size),
                compression="adobe_deflate",
                predictor=True,
            )
        if verbose:
            print()

