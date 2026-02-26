"""
Interpolation utilities for sparse sample maps.

Functions here convert a sparse dict mapping ``(x, y) -> label`` to a
dense array via several strategies:

- :func:`interpolate_edt`        – binary foreground/background EDT
- :func:`interpolate_multiclass` – multi-class Voronoi via single EDT pass
- :func:`interpolate_linear`     – continuous values via scipy griddata
"""
from typing import Dict, Tuple, Optional, List, Union

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import griddata


def interpolate_edt(
    samples: Dict[Tuple[int, int], int],
    height: int,
    width: int,
    cluster_idx: Union[int, List[int]],
    downsample: int = 1,
    upsample: bool = True,
) -> np.ndarray:
    """
    Interpolate binary mask using Euclidean Distance Transform.

    Parameters
    ----------
    samples : Dict[Tuple[int, int], int]
        Dictionary mapping (x, y) coordinates to cluster labels.
    height : int
        Height of output image.
    width : int
        Width of output image.
    cluster_idx : int or list of int
        Target cluster index(es) for foreground.
    downsample : int, default=1
        Factor by which to downsample before interpolation.
    upsample : bool, default=True
        Whether to upsample result back to original resolution.

    Returns
    -------
    interpolated : np.ndarray
        Binary mask with interpolated values.
    """
    if isinstance(cluster_idx, int):
        cluster_idx = [cluster_idx]

    full_height, full_width = height, width
    height, width = height // downsample, width // downsample

    canvas_foreground = np.zeros((height, width), dtype=bool)
    canvas_background = np.zeros((height, width), dtype=bool)

    for (x, y), value in samples.items():
        x, y = x // downsample, y // downsample
        x = np.clip(x, 0, width - 1)
        y = np.clip(y, 0, height - 1)
        if value in cluster_idx:
            canvas_foreground[y, x] = True
        else:
            canvas_background[y, x] = True

    edt_foreground = distance_transform_edt(~canvas_background)
    edt_background = distance_transform_edt(~canvas_foreground)
    interpolated = edt_foreground > edt_background

    if downsample > 1 and upsample:
        interpolated = cv2.resize(
            interpolated.astype(np.uint8),
            (full_width, full_height),
            interpolation=cv2.INTER_NEAREST_EXACT,
        ).astype(bool)

    return interpolated


def interpolate_multiclass(
    samples: Dict[Tuple[int, int], int],
    height: int,
    width: int,
    downsample: int = 1,
    upsample: bool = True,
    background_label: int = -1,
) -> np.ndarray:
    """
    Interpolate multi-class label map using nearest-neighbour Voronoi
    assignment via a single EDT index lookup.

    This is O(H*W) regardless of the number of classes, since
    ``distance_transform_edt(..., return_indices=True)`` gives the index of
    the nearest sampled point for every pixel in one pass.

    Parameters
    ----------
    samples : Dict[Tuple[int, int], int]
        Dictionary mapping (x, y) coordinates to cluster labels.
    height, width : int
        Full-resolution output dimensions.
    downsample : int, default=1
        Downsample factor before interpolation.
    upsample : bool, default=True
        Upsample result back to (height, width) after interpolation.
    background_label : int, default=-1
        Label used for unsampled positions; these are treated as "no sample"
        seeds and will be overwritten by the nearest real sample.

    Returns
    -------
    label_map : np.ndarray, shape (height, width), dtype int32
        Dense label array.
    """
    full_height, full_width = height, width
    height = height // downsample
    width  = width  // downsample

    seed_mask    = np.zeros((height, width), dtype=bool)
    label_canvas = np.full((height, width), background_label, dtype=np.int32)

    for (x, y), value in samples.items():
        xi = int(np.clip(x // downsample, 0, width  - 1))
        yi = int(np.clip(y // downsample, 0, height - 1))
        seed_mask[yi, xi]    = True
        label_canvas[yi, xi] = value

    # Single EDT call: nearest_idx[0/1] gives the row/col of the closest seed
    # for every pixel — O(H*W).
    _, nearest_idx = distance_transform_edt(~seed_mask, return_indices=True)
    label_map = label_canvas[nearest_idx[0], nearest_idx[1]]

    if downsample > 1 and upsample:
        label_map = cv2.resize(
            label_map.astype(np.int32),
            (full_width, full_height),
            interpolation=cv2.INTER_NEAREST_EXACT,
        )

    return label_map


def interpolate_linear(
    samples: Dict[Tuple[int, int], int],
    height: int,
    width: int,
    downsample: int = 1,
    mapping: Optional[Dict[int, int]] = None,
) -> np.ndarray:
    """
    Interpolate continuous values using linear griddata interpolation.

    Parameters
    ----------
    samples : Dict[Tuple[int, int], int]
        Dictionary mapping (x, y) coordinates to cluster labels.
    height : int
        Height of output image.
    width : int
        Width of output image.
    downsample : int, default=1
        Factor by which to downsample the output.
    mapping : dict, optional
        Dictionary to remap cluster labels before interpolation.

    Returns
    -------
    interpolated : np.ndarray
        Continuously interpolated values in downsampled resolution.
    """
    points = np.array(list(samples.keys()), dtype=float)
    values = list(samples.values())

    if mapping is not None:
        values = [mapping[v] for v in values]
    values = np.array(values, dtype=float)

    out_height = height // downsample
    out_width  = width  // downsample

    # Convert (x, y) sample coords to downsampled (y, x) order for griddata
    points_down = points / downsample
    points_down = points_down[:, [1, 0]]  # (x, y) -> (y, x)

    grid_y, grid_x = np.mgrid[0:out_height, 0:out_width]

    interpolated = griddata(
        points_down, values, (grid_y, grid_x),
        method='linear', fill_value=0.0,
    )

    return interpolated
