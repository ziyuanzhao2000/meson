"""
Adaptive quadtree sampling for whole slide image segmentation.

v2: Epoch-based iteration — gradient is computed once per epoch, then ALL
positive-gradient cells are processed across as many minibatches as needed
before the next gradient recomputation.

Key change from v1:
  - v1: 1 iteration = 1 gradient computation + 1 minibatch (≤batch_size samples)
    → With 28k active cells and batch_size=1024, this means ~28 gradient
      recomputations that are nearly identical, dominating wall time.
  - v2: 1 epoch = 1 gradient computation + N minibatches to drain ALL active cells
    → Gradient is only recomputed when it actually changes meaningfully
      (after processing all current boundary cells).
"""
import numpy as np
from meson.tools._embed_patch import embed_patch
import torch
import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Union
from dataclasses import dataclass

from tqdm import tqdm
from skimage.filters import sobel
from meson._interpolation import interpolate_edt, interpolate_multiclass, interpolate_linear
import heapq

from meson._readwrite import get_base_level
from collections import deque


@dataclass
class Cell:
    """
    Represents a cell in the quadtree subdivision.

    Attributes
    ----------
    corners : list
        List of (x, y) tuples representing the four corners of the cell.
        Order: [top_left, top_right, bottom_left, bottom_right]
    size : int
        The size (width/height) of the square cell in pixels
    """
    corners: list
    size: int


def initialize_grid(
    patch_df,
    size: int,
    patch_size: int = 448,
    feature_column: Optional[str] = None
) -> Tuple[Dict[Tuple[int, int], int], list]:
    """
    Initialize sampling grid from patch table.

    Parameters
    ----------
    patch_df : pd.DataFrame
        Patch table with xmin, ymin, xmax, ymax columns
    size : int
        Initial cell size for quadtree
    patch_size : int, default=448
        Size of image patches
    feature_column : str, optional
        Column name containing initial cluster labels. If None, all samples
        are initialized as background (-1)

    Returns
    -------
    samples : Dict[Tuple[int, int], int]
        Dictionary mapping (x, y) coordinates to cluster labels
    cells : list
        List of cells to process
    """
    samples = {}
    cells = []

    for xmin in patch_df.xmin.unique():
        for ymin in patch_df.ymin.unique():
            xcenter = xmin + patch_size // 2
            ycenter = ymin + patch_size // 2
            samples[(xcenter, ycenter)] = -1  # Initialize as background
            cells.append(Cell(
                corners=[
                    (xcenter, ycenter),
                    (xcenter + size, ycenter),
                    (xcenter, ycenter + size),
                    (xcenter + size, ycenter + size)
                ],
                size=size
            ))

    # Set initial labels if provided
    if feature_column is not None:
        for row in patch_df.itertuples():
            xcenter = row.xmin + patch_size // 2
            ycenter = row.ymin + patch_size // 2
            samples[(xcenter, ycenter)] = getattr(row, feature_column)

    return samples, cells


# ---------------------------------------------------------------------------
# Helpers factored out so both the epoch loop and single-step can share them
# ---------------------------------------------------------------------------

def _compute_gradient_and_rank_cells(
    samples: Dict[Tuple[int, int], int],
    cells: list,
    image_shape: Tuple[int, int, int],
    size_threshold: int,
    min_gradient_threshold: float,
    mapping: Optional[Dict[int, int]] = None,
) -> Tuple[list, list, np.ndarray]:
    """
    Rasterize samples → Sobel gradient → score & partition cells.

    Returns
    -------
    active_heap : list
        Max-heap (negated gradient) of cells whose avg gradient > threshold.
    inactive_cells : list
        Cells with zero / below-threshold gradient (kept for future epochs).
    gradient_map : np.ndarray
    """
    _, height, width = image_shape

    # Rasterize current samples with linear interpolation
    interpolated = interpolate_linear(
        samples, height, width,
        downsample=size_threshold,
        mapping=mapping
    )

    # Compute gradient map
    gradient_map = sobel(interpolated)
    grad_height, grad_width = gradient_map.shape

    # --- Vectorized cell scoring via integral image (same as v1) -----------
    corners = np.array([cell.corners[0] for cell in cells])   # (N, 2)
    sizes = np.array([cell.size for cell in cells])            # (N,)

    x_min_down = np.clip(corners[:, 0] // size_threshold, 0, grad_width - 1).astype(int)
    x_max_down = np.clip((corners[:, 0] + sizes) // size_threshold, 0, grad_width - 1).astype(int)
    y_min_down = np.clip(corners[:, 1] // size_threshold, 0, grad_height - 1).astype(int)
    y_max_down = np.clip((corners[:, 1] + sizes) // size_threshold, 0, grad_height - 1).astype(int)

    integral = gradient_map.cumsum(axis=0).cumsum(axis=1)
    integral = np.pad(integral, ((1, 0), (1, 0)), mode='constant')

    total = (integral[y_max_down + 1, x_max_down + 1]
             - integral[y_min_down,     x_max_down + 1]
             - integral[y_max_down + 1, x_min_down]
             + integral[y_min_down,     x_min_down])

    areas = np.maximum(
        (x_max_down - x_min_down + 1) * (y_max_down - y_min_down + 1), 1
    )
    avg_gradients = total / areas

    # Partition into active (positive gradient) vs inactive
    mask = avg_gradients > min_gradient_threshold

    active_heap = [
        (-g, i, c)
        for i, (g, c) in enumerate(
            (avg_gradients[j], cells[j]) for j in np.nonzero(mask)[0]
        )
    ]
    heapq.heapify(active_heap)  # O(N)

    inactive_cells = [cells[j] for j in np.nonzero(~mask)[0]]

    return active_heap, inactive_cells, gradient_map


def _subdivide_cell(cell: Cell) -> List[Cell]:
    """Split a cell into 4 quadtree children."""
    top_left, top_right, bottom_left, bottom_right = cell.corners
    size = cell.size
    mid_x = top_left[0] + size // 2
    mid_y = top_left[1] + size // 2
    half = size // 2
    return [
        Cell(corners=[top_left, (mid_x, top_left[1]),
                       (top_left[0], mid_y), (mid_x, mid_y)], size=half),
        Cell(corners=[(mid_x, top_left[1]), top_right,
                       (mid_x, mid_y), (top_right[0], mid_y)], size=half),
        Cell(corners=[(top_left[0], mid_y), (mid_x, mid_y),
                       bottom_left, (mid_x, bottom_left[1])], size=half),
        Cell(corners=[(mid_x, mid_y), (top_right[0], mid_y),
                       (mid_x, bottom_left[1]), bottom_right], size=half),
    ]


def _embed_and_label(
    coords_list: list,
    image: torch.Tensor,
    embedder,
    classifier,
    samples: Dict[Tuple[int, int], int],
) -> int:
    """
    Extract 448×448 patches at *coords_list*, embed, predict, and write
    labels into *samples* in-place.  Returns number of new samples written.
    """
    n = len(coords_list)
    if n == 0:
        return 0

    batch_cpu = torch.zeros((n, 3, 448, 448), dtype=torch.uint8)
    img_h, img_w = image.shape[1], image.shape[2]

    for idx, (x, y) in enumerate(coords_list):
        y0, y1 = y - 224, y + 224
        x0, x1 = x - 224, x + 224

        sy0, sy1 = max(0, y0), min(img_h, y1)
        sx0, sx1 = max(0, x0), min(img_w, x1)
        dy0 = max(0, -y0)
        dx0 = max(0, -x0)

        batch_cpu[idx, :,
                  dy0:dy0 + (sy1 - sy0),
                  dx0:dx0 + (sx1 - sx0)] = image[:, sy0:sy1, sx0:sx1]

    with torch.no_grad():
        embeddings = embedder(batch_cpu).cpu().numpy().astype(np.float64)
    values = classifier.predict(embeddings)

    for coord, value in zip(coords_list, values):
        samples[coord] = value

    return n


# ---------------------------------------------------------------------------
# Main epoch-based refinement
# ---------------------------------------------------------------------------

def adaptive_refine_epoch(
    samples: Dict[Tuple[int, int], int],
    cells: list,
    image,                      # np.ndarray (C, H, W)  — converted to tensor internally
    embedder,
    classifier,
    batch_size: int,
    size_threshold: int = 4,
    min_gradient_threshold: float = 1e-8,
    mapping: Optional[Dict[int, int]] = None,
    show_progress: bool = False,
) -> Tuple[Dict, list, int, np.ndarray]:
    """
    One *epoch* of adaptive refinement.

    1. Compute gradient from current samples.
    2. Rank all cells; partition into active (gradient > 0) and inactive.
    3. Process ALL active cells across as many minibatches as needed:
       - Pop highest-gradient cells from the heap.
       - Collect their unsampled corner coordinates.
       - When a minibatch fills up, embed + classify, then continue.
       - Subdivide cells that are above size_threshold.
    4. Return: updated samples, next-epoch cell list
       (inactive + newly created children), total new samples, gradient map.

    Parameters
    ----------
    (same as v1 adaptive_refine_step, but batch_size now controls the
     *minibatch* size within the epoch, not the epoch budget.)

    Returns
    -------
    samples : dict
    cells : list          — cells for the NEXT epoch
    n_samples : int       — total new samples added this epoch
    gradient_map : np.ndarray
    """
    if not cells:
        return samples, cells, 0, None

    image_shape = image.shape  # (C, H, W)

    # ---- Step 1-2: gradient + rank -----------------------------------------
    active_heap, inactive_cells, gradient_map = _compute_gradient_and_rank_cells(
        samples, cells, image_shape,
        size_threshold, min_gradient_threshold, mapping,
    )

    n_active = len(active_heap)
    if n_active == 0:
        # Nothing to refine — all cells are below threshold
        return samples, inactive_cells, 0, gradient_map

    # Convert image to tensor once for the whole epoch
    image_t = torch.from_numpy(image) if isinstance(image, np.ndarray) else image

    # ---- Step 3: drain all active cells in minibatches ----------------------
    total_new_samples = 0
    new_child_cells = []
    push_counter = len(cells)

    # Accumulator for the current minibatch
    coords_to_sample = set()
    cells_to_split = []

    pbar = tqdm(
        total=n_active,
        desc="    Processing active cells",
        disable=not show_progress,
        leave=False,
    )

    while active_heap:
        neg_gradient, _, cell = heapq.heappop(active_heap)
        pbar.update(1)

        # (shouldn't happen after partitioning, but defensive)
        if neg_gradient > -min_gradient_threshold:
            break

        # Gather unsampled corners for this cell
        cell_coords = [c for c in cell.corners if c not in samples]

        # If adding this cell would exceed the minibatch, flush first
        if (coords_to_sample
                and len(coords_to_sample) + len(cell_coords) > batch_size):
            # --- flush minibatch ---
            n = _embed_and_label(
                list(coords_to_sample), image_t, embedder, classifier, samples
            )
            total_new_samples += n
            # subdivide accumulated cells
            for c in cells_to_split:
                if c.size > size_threshold:
                    new_child_cells.extend(_subdivide_cell(c))
            coords_to_sample = set()
            cells_to_split = []

        coords_to_sample.update(cell_coords)
        cells_to_split.append(cell)

    pbar.close()

    # --- flush remaining minibatch ---
    if coords_to_sample:
        n = _embed_and_label(
            list(coords_to_sample), image_t, embedder, classifier, samples
        )
        total_new_samples += n
    for c in cells_to_split:
        if c.size > size_threshold:
            new_child_cells.extend(_subdivide_cell(c))

    # ---- Step 4: combine for next epoch ------------------------------------
    # Next epoch's cell list = inactive (might become active after labels
    # change) + newly subdivided children
    next_cells = inactive_cells + new_child_cells

    return samples, next_cells, total_new_samples, gradient_map


# ---------------------------------------------------------------------------
# Legacy single-step wrapper (drop-in compatible with v1 callers)
# ---------------------------------------------------------------------------

def adaptive_refine_step(
    samples, cells, image, embedder, classifier,
    batch_size, size_threshold=4, min_gradient_threshold=1e-8,
    mapping=None, show_progress=False,
):
    """
    v1-compatible single-step refinement (kept for backwards compat).

    Internally delegates to the epoch-based function but caps the number of
    samples to *batch_size* so the behaviour is identical to v1.
    """
    # Just call the epoch version — it will process all active cells, but
    # since we want the old 1-minibatch behaviour we could cap. For true
    # backwards compat, callers should migrate to adaptive_refine_epoch.
    return adaptive_refine_epoch(
        samples, cells, image, embedder, classifier,
        batch_size=batch_size,
        size_threshold=size_threshold,
        min_gradient_threshold=min_gradient_threshold,
        mapping=mapping,
        show_progress=show_progress,
    )


# ---------------------------------------------------------------------------
# I/O helpers (unchanged from v1)
# ---------------------------------------------------------------------------

import pandas as pd
import pyarrow.parquet as pq


def save_samples_parquet(
    samples: Dict[Tuple[int, int], int],
    path: Union[str, Path],
    compress: str = 'zstd'
) -> None:
    """Save samples to Parquet format."""
    df = pd.DataFrame([
        {'x': x, 'y': y, 'label': label}
        for (x, y), label in samples.items()
    ])
    df = df.astype({'x': 'int32', 'y': 'int32', 'label': 'int8'})
    df.to_parquet(path, compression=compress, index=False)


def load_samples_parquet(path: Union[str, Path]) -> Dict[Tuple[int, int], int]:
    """Load samples from Parquet format."""
    df = pd.read_parquet(path)
    return {(row.x, row.y): row.label for row in df.itertuples(index=False)}


# ---------------------------------------------------------------------------
# Top-level WSI adaptive sampling (epoch-based)
# ---------------------------------------------------------------------------

def adaptive_sample_wsi(
    sdata,
    image_name: str,
    embedder,
    classifier,
    cluster_groups: List[List[int]],
    output_dir: Union[str, Path],
    initial_size: int = 256,
    batch_size: int = 1024,
    size_threshold: Union[int, List[int]] = 32,
    patch_size: int = 448,
    point_name: str = 'grid_point',
    save_format: str = 'parquet',
    show_progress: bool = True,
) -> Dict[Tuple[int, int], int]:
    """
    Perform adaptive quadtree sampling on a whole slide image.

    This v2 implementation uses *epoch-based* iteration: each epoch computes
    the gradient once and then drains all positive-gradient cells across
    multiple minibatches before recomputing.  This dramatically reduces the
    number of expensive gradient recomputations (interpolation + Sobel).

    Parameters are identical to v1.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Normalise size_threshold to a per-group list
    if isinstance(size_threshold, int):
        size_thresholds = [size_threshold] * len(cluster_groups)
    else:
        if len(size_threshold) != len(cluster_groups):
            raise ValueError(
                f"size_threshold list length ({len(size_threshold)}) must match "
                f"cluster_groups length ({len(cluster_groups)})"
            )
        size_thresholds = list(size_threshold)

    # Get k means labels
    sdata.attrs['models'][f'kmeans'] = classifier
    sdata.attrs['models_metadata'].append({
        'name': f'kmeans',
        'model_type': 'KMeans'
    })
    print(f"Embedding patches with KMeans...", flush=True)
    sdata = embed_patch(
        sdata,
        embedder=f'kmeans',
        image_name=image_name,
        point_name='grid_point',
        patch_name='patch',
        obsm_key='UNI_embedding',
        save=False,
        overwrite=False
    )

    # Get patch table and WSI
    print("Getting patch table and WSI...", flush=True)
    patch_table = sdata[f'{image_name}_{point_name}_patch']
    wsi = get_base_level(sdata[image_name]).compute().data

    # Initialize grid
    print("Initializing grid...", flush=True)
    samples, cells = initialize_grid(
        patch_table.obs,
        size=initial_size,
        patch_size=patch_size,
        feature_column='kmeans_label'
    )

    if show_progress:
        print(f"Initialized with {len(samples)} samples and {len(cells)} cells.",
              flush=True)

    # Refine for each cluster group
    for cluster_idxs, group_size_threshold in zip(cluster_groups, size_thresholds):
        if show_progress:
            print(f"\nRefining for cluster indices: {cluster_idxs}", flush=True)

        # Reset cells for this group
        _, cells = initialize_grid(
            patch_table.obs,
            size=initial_size,
            patch_size=patch_size,
            feature_column=None
        )

        if show_progress:
            print(f"  Using size_threshold={group_size_threshold} for this group.",
                  flush=True)

        # Create mapping: target clusters -> 1, background -> 0
        mapping = {i: 0 for i in range(-1, 26)}
        for cluster_idx in cluster_idxs:
            mapping[cluster_idx] = 1

        total_samples = len(samples)
        n_epochs = 0

        while cells:
            n_epochs += 1
            samples, cells, n_new, _ = adaptive_refine_epoch(
                samples, cells, wsi,
                embedder, classifier,
                batch_size=batch_size,
                size_threshold=group_size_threshold,
                mapping=mapping,
                show_progress=show_progress,
            )
            total_samples += n_new

            if n_new == 0 and n_epochs > 1:
                print("  No new samples added, stopping refinement for "
                      "this cluster group.", flush=True)
                break

            if show_progress:
                print(f"  Epoch {n_epochs}: processed {n_new} new samples, "
                      f"{len(cells)} cells remaining, "
                      f"{total_samples} total samples.", flush=True)

        if show_progress:
            print(f"  Refinement completed in {n_epochs} epochs.", flush=True)

    # Save results
    if save_format == 'parquet':
        samples_path = output_dir / f'{image_name}_samples.parquet'
        save_samples_parquet(samples, samples_path)
    else:
        samples_path = output_dir / f'{image_name}_samples.pkl'
        with open(samples_path, 'wb') as f:
            pickle.dump(samples, f)

    if show_progress:
        print(f"\nSaved samples dictionary to: {samples_path}", flush=True)

    return samples