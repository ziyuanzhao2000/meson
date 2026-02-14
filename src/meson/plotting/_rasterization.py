from typing import Dict, Tuple, Deque

import cv2
from matplotlib.table import Cell
from scipy.ndimage import distance_transform_edt
import numpy as np

def interpolate_multiclass(
    samples: Dict[Tuple[int, int], int],
    height: int,
    width: int,
    downsample: int,
    upsample: bool = True,
) -> np.ndarray:
    """
    Interpolate a multi-class label map from sparse samples using
    nearest-neighbor assignment via Euclidean Distance Transform.

    Each pixel is assigned the class whose nearest sample point is closest
    (i.e., Voronoi tessellation of the sample points).
    """
    full_height, full_width = height, width
    height, width = height // downsample, width // downsample

    classes = set(samples.values())

    # For each class, compute EDT from all *other* classes' points.
    # A pixel belongs to class c if it's farther from non-c points
    # than from c points — equivalently, if edt_others[c] is minimal.
    # But it's simpler: compute EDT from c's own points, then argmin.
    edts = {}
    for c in classes:
        canvas = np.ones((height, width), dtype=bool)  # True everywhere = "empty"
        for (x, y), value in samples.items():
            if value == c:
                x, y = x // downsample, y // downsample
                x = np.clip(x, 0, width - 1)
                y = np.clip(y, 0, height - 1)
                canvas[y, x] = False  # Mark as occupied
        edts[c] = distance_transform_edt(canvas)

    # Stack and argmin → each pixel gets the class with the nearest sample
    class_list = sorted(classes)
    edt_stack = np.stack([edts[c] for c in class_list], axis=0)  # (C, H, W)
    label_indices = np.argmin(edt_stack, axis=0)  # (H, W)

    # Map indices back to original class labels
    class_arr = np.array(class_list)
    interpolated = class_arr[label_indices]

    if downsample > 1 and upsample:
        interpolated = cv2.resize(
            interpolated.astype(np.int32),
            (full_width, full_height),
            interpolation=cv2.INTER_NEAREST_EXACT,
        )

    return interpolated
    
def extract_samples(df, column, patch_size = 448):
    samples = {}
    for xmin in df.xmin.unique():
        for ymin in df.ymin.unique():
            xcenter = int(xmin + patch_size // 2)
            ycenter = int(ymin + patch_size // 2)
            samples[(xcenter, ycenter)] = -1 # initialize all samples as background
    for row in df.itertuples():
        xmin, xmax, ymin, ymax = row.xmin, row.xmax, row.ymin, row.ymax
        xcenter = int(xmin + patch_size // 2)
        ycenter = int(ymin + patch_size // 2)
        samples[(xcenter, ycenter)] = getattr(row, column)
        
    return samples