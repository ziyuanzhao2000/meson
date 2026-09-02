from typing import TYPE_CHECKING

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from meson._readwrite import get_base_level, overwrite_element
from meson._utils import get_optimal_chunk_size
from tqdm import tqdm
from scipy.sparse import csc_array, hstack
import pandas as pd
from PIL import Image


if TYPE_CHECKING:
    from spatialdata import SpatialData

def _get_embedder(sdata, embedder_name: str, token=None):
    """Convert embedder name to embedder instance"""
    if embedder_name == "test":
        from meson.tools.embedders import TestEmbedder
        return TestEmbedder(), "Vision"
    elif embedder_name in ["uni", "UNI"]:
        from meson.tools.embedders import UNIEmbedder
        return UNIEmbedder(token=token), "Vision"
    elif embedder_name in ["uni2", "UNI2", "uni2-h", "UNI2-h"]:
        from meson.tools.embedders import UNI2Embedder
        return UNI2Embedder(token=token), "Vision"
    elif embedder_name in ["kmeans", "KMeans", "FrequencyRankedKMeans"]:
        from meson.tools.embedders import FrequencyRankedKMeans
        for model_info in sdata.attrs['models_metadata']:
            model_name = model_info['name']
            if embedder_name == model_name:
                return sdata.attrs['models'][model_name], "KMeans"
        raise ValueError(f"KMeans model '{embedder_name}' not found in sdata.attrs['models']")
    else:
        for model_info in sdata.attrs['models_metadata']:
            model_name = model_info['name']
            if embedder_name == model_name:
                return sdata.attrs['models'][model_name], model_info['model_type']
        raise ValueError(f"Unknown embedder: {embedder_name}")


# --------------------------------------------------------------------------- #
# Helpers for block-wise patch reading (Vision branch)
# --------------------------------------------------------------------------- #

def _as_numpy(block, scheduler=None):
    """Materialise a possibly-lazy block (xarray DataArray / dask array) as numpy.

    Handles all three things `PatchDataset.image` can be -- a plain numpy array
    (precache), a chunked xarray DataArray, or a bare dask array -- so the callers
    below never have to care which one they got.
    """
    if isinstance(block, np.ndarray):
        return block
    if hasattr(block, "compute"):
        block = block.compute(**({"scheduler": scheduler} if scheduler else {}))
    if isinstance(block, np.ndarray):
        return block
    return np.asarray(getattr(block, "data", block))


def _uniform_patch_geometry(patch_df, image_shape):
    """Return (patch_h, patch_w) if every patch is the same size and in bounds.

    Block reading stacks the patches of a block into one array, which requires a
    single shape. Returns None when the grid is ragged or runs off the edge of the
    image, which tells the caller to fall back to the original per-patch path
    instead of failing.
    """
    try:
        ymin = patch_df['ymin'].to_numpy()
        ymax = patch_df['ymax'].to_numpy()
        xmin = patch_df['xmin'].to_numpy()
        xmax = patch_df['xmax'].to_numpy()
    except KeyError:
        return None
    if len(ymin) == 0:
        return None

    heights, widths = ymax - ymin, xmax - xmin
    if heights.min() != heights.max() or widths.min() != widths.max():
        return None
    if ymin.min() < 0 or xmin.min() < 0:
        return None
    if ymax.max() > image_shape[-2] or xmax.max() > image_shape[-1]:
        return None
    return int(heights[0]), int(widths[0])


def _aligned_block_size(image, patch_hw, target_px):
    """Pick a (height, width) read block, snapped to the dask chunk grid.

    Reading on the chunk grid means each compressed chunk is decoded exactly once
    for the whole block instead of once per patch that overlaps it. The block is
    never smaller than one patch, and alignment is skipped if the chunks are so
    large that snapping would blow up the block (and hence peak memory).
    """
    patch_h, patch_w = patch_hw
    chunk_y = chunk_x = None
    try:
        chunks = getattr(image, "chunks", None)
        if chunks is not None:
            chunk_y = int(max(chunks[-2]))
            chunk_x = int(max(chunks[-1]))
    except Exception:
        chunk_y = chunk_x = None

    def _snap(chunk, minimum):
        if not chunk or chunk > 2 * target_px:
            return max(target_px, minimum)
        return max(max(1, int(round(target_px / chunk))) * chunk, minimum)

    return _snap(chunk_y, patch_h), _snap(chunk_x, patch_w)


def _group_patches_into_blocks(patch_df, block_h, block_w):
    """Bucket patch rows into spatial blocks, one read per block.

    Each entry is (row_indices, y0, y1, x0, x1) where the bounds are the bounding
    box of that bucket's patches. Buckets come out in row-major order so
    consecutive reads are close together on disk.
    """
    ymin = patch_df['ymin'].to_numpy().astype(np.int64)
    ymax = patch_df['ymax'].to_numpy().astype(np.int64)
    xmin = patch_df['xmin'].to_numpy().astype(np.int64)
    xmax = patch_df['xmax'].to_numpy().astype(np.int64)

    by, bx = ymin // block_h, xmin // block_w
    key = by * (int(bx.max()) + 1) + bx

    order = np.argsort(key, kind="stable")
    cuts = np.flatnonzero(np.diff(key[order])) + 1

    blocks = []
    for rows in np.split(order, cuts):
        blocks.append((rows,
                       int(ymin[rows].min()), int(ymax[rows].max()),
                       int(xmin[rows].min()), int(xmax[rows].max())))
    return blocks


class PatchBlockDataset(Dataset):
    """One item per spatial block: all of that block's patches, already stacked.

    This is the memory/speed fix for the Vision branch. The original dataset
    computed one dask slice per patch, so a chunk was decompressed once for every
    patch touching it, and the only way out was `precache=True` (whole slide in
    RAM). Here a block is read once, patches are cut out of it with plain numpy
    slicing, and the block is freed as soon as the item is returned -- so peak
    memory is a few blocks, not the slide.

    Returns (row_indices, uint8-or-native tensor of shape (n, C, patch_h, patch_w)).
    Row indices are carried through explicitly so the output rows stay correct
    regardless of block ordering.
    """

    def __init__(self, image, patch_df, blocks, patch_hw, color_transform=None,
                 scheduler=None):
        self.image = image
        self.blocks = blocks
        self.patch_h, self.patch_w = patch_hw
        self.color_transform = color_transform
        self.scheduler = scheduler
        self.ymin = patch_df['ymin'].to_numpy().astype(np.int64)
        self.xmin = patch_df['xmin'].to_numpy().astype(np.int64)

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, i):
        rows, y0, y1, x0, x1 = self.blocks[i]
        block = _as_numpy(self.image[:, y0:y1, x0:x1], scheduler=self.scheduler)

        ys = self.ymin[rows] - y0
        xs = self.xmin[rows] - x0
        ph, pw = self.patch_h, self.patch_w

        if self.color_transform is not None:
            # Same semantics as the original: CHW -> PIL -> transform -> CHW.
            patches = np.stack([
                np.array(self.color_transform(
                    Image.fromarray(block[:, y:y + ph, x:x + pw].transpose(1, 2, 0))
                )).transpose(2, 0, 1)
                for y, x in zip(ys, xs)
            ])
        else:
            patches = np.empty((len(rows), block.shape[0], ph, pw), dtype=block.dtype)
            for j, (y, x) in enumerate(zip(ys, xs)):
                patches[j] = block[:, y:y + ph, x:x + pw]

        del block
        return rows, torch.from_numpy(np.ascontiguousarray(patches))


class PatchDataset(Dataset):
    def __init__(self, sdata: "SpatialData", 
                        image_name: str | None = None,
                        point_name: str | None = 'grid_point',
                        patch_name: str | None = 'patch', 
                        precache: bool = False,
                        color_transform = None):
        self.precache = precache
        if image_name is not None:
            patch_name = f'{image_name}_{point_name}_{patch_name}'
        image = get_base_level(sdata[image_name])
        if self.precache:
            self.image = image.compute().data
        else:
            self.image = image.chunk(chunks=get_optimal_chunk_size(image))
        print("Initialized patch dataset for image with shape", self.image.shape)
        self.patch_df = sdata[patch_name].obs

        self.color_transform = color_transform

    def __getitem__(self, index):
        row = self.patch_df.iloc[index]
        patch = self.image[
            :,  # All channels
            row['ymin']:row['ymax'],
            row['xmin']:row['xmax']
        ]
        if not self.precache:
            patch = patch.compute().data
    
        if not self.color_transform is None:
            patch = Image.fromarray(patch.transpose(1,2,0))
            return np.array(self.color_transform(patch)).transpose(2, 0, 1)
        else:
            return patch
    def __len__(self):
        return len(self.patch_df)


class _IndexedPatchDataset(PatchDataset):
    """Fallback path: the original per-patch dataset, but tagged with row indices.

    Used only when the patch grid is ragged or out of bounds, so block stacking is
    impossible. Emitting (row_indices, patches) in the same shape as
    PatchBlockDataset lets both paths share one inference loop.
    """

    def __getitem__(self, index):
        patch = super().__getitem__(index)
        patch = torch.from_numpy(np.ascontiguousarray(patch))
        return np.array([index]), patch[None]


def _identity_collate(batch):
    """One block per item already; module-level so it survives worker pickling."""
    return batch[0]


def _iter_model_batches(loader, batch_size):
    """Regroup variable-sized per-block stacks into fixed-size model batches.

    Decouples "how many patches happen to sit in this block" from "how many patches
    the GPU wants at once", so a sparse block never causes a tiny, GPU-starving
    forward pass and a dense one never overflows the requested batch size.
    """
    rows_buf, patch_buf, pending = [], [], 0
    for rows, patches in loader:
        rows_buf.append(rows)
        patch_buf.append(patches)
        pending += len(rows)
        if pending >= batch_size:
            rows_all = np.concatenate(rows_buf)
            patches_all = torch.cat(patch_buf)
            cut = (pending // batch_size) * batch_size
            for s in range(0, cut, batch_size):
                yield rows_all[s:s + batch_size], patches_all[s:s + batch_size]
            rows_buf, patch_buf = [rows_all[cut:]], [patches_all[cut:]]
            pending -= cut
    if pending:
        yield np.concatenate(rows_buf), torch.cat(patch_buf)


def _embed_patches_vision(sdata, embedder, *, image_name, point_name, patch_name,
                          patch_table, batch_size, num_workers, device, precache,
                          color_transform, block_px, embedding_dtype, use_amp,
                          pin_memory):
    """Run the vision foundation model over every patch and return the embeddings.

    Block-reads the image where the patch grid allows it (the fast path) and falls
    back to the original per-patch reader otherwise. Output is preallocated at
    `embedding_dtype` (float32 by default) rather than float64, which halves the
    memory the embedding matrix occupies for the rest of the session.
    """
    base_dataset = PatchDataset(sdata, image_name, point_name, patch_name,
                                precache=precache, color_transform=color_transform)
    n_patches = len(base_dataset)
    patch_hw = _uniform_patch_geometry(base_dataset.patch_df, base_dataset.image.shape)

    if patch_hw is not None:
        block_h, block_w = _aligned_block_size(base_dataset.image, patch_hw, block_px)
        blocks = _group_patches_into_blocks(base_dataset.patch_df, block_h, block_w)
        print(f"Reading {n_patches} patches of {patch_hw[0]}x{patch_hw[1]}px in "
              f"{len(blocks)} blocks of up to {block_h}x{block_w}px")
        dataset = PatchBlockDataset(
            base_dataset.image, base_dataset.patch_df, blocks, patch_hw,
            color_transform=color_transform,
            # Workers already give parallelism; a dask thread pool inside each one
            # would only oversubscribe the CPUs.
            scheduler="synchronous" if (num_workers and not precache) else None,
        )
    else:
        print("Patch grid is ragged or out of bounds; using the per-patch reader.")
        dataset = _IndexedPatchDataset(sdata, image_name, point_name, patch_name,
                                       precache=precache, color_transform=color_transform)

    loader = DataLoader(dataset,
                        batch_size=1,
                        shuffle=False,
                        num_workers=num_workers,
                        collate_fn=_identity_collate,
                        pin_memory=pin_memory and str(device) != 'cpu',
                        persistent_workers=num_workers > 0,
                        prefetch_factor=2 if num_workers > 0 else None)

    embedding_array = np.zeros((n_patches, embedder.embed_dim), dtype=embedding_dtype)
    amp_on = bool(use_amp) and 'cuda' in str(device)
    n_batches = (n_patches + batch_size - 1) // batch_size

    with torch.inference_mode():
        for rows, batch in tqdm(_iter_model_batches(loader, batch_size),
                                total=n_batches, desc="Embedding patches"):
            batch = batch.to(device, non_blocking=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp_on):
                batch_embedding = embedder(batch)
            embedding_array[rows] = batch_embedding.float().cpu().numpy()

    del loader, dataset, base_dataset
    return embedding_array


def embed_patch(sdata: "SpatialData",
                embedder,
                *,
                image_name: str | None = None,
                point_name: str | None = 'grid_point',
                patch_name: str | None = 'patch',
                batch_size: int = 32, 
                num_workers: int = 1,
                token: str | None = None,
                obsm_key: str | None = None,
                device: str = 'cpu',
                save: bool = True,
                overwrite: bool = False,
                precache: bool = False, 
                color_transform = None,
                column_keep_indices: list[int] | None = None,
                block_px: int = 4096,
                embedding_dtype = np.float32,
                use_amp: bool = False,
                pin_memory: bool = True):
    if image_name is not None:
        patch_name_full = f'{image_name}_{point_name}_{patch_name}'
    else:
        patch_name_full = patch_name
    patch_table = sdata.tables[patch_name_full]
    if isinstance(embedder, str):
        embedder_name = embedder
        embedder, model_type = _get_embedder(sdata, embedder, token=token)
    else:
        embedder_name = embedder.name
    
    if model_type == "Vision":
        embedding_name = f'{embedder.name}_embedding'
        if overwrite is False and embedding_name in patch_table.obsm:
            return sdata
        if hasattr(embedder, 'model'):
            embedder.model.to(device)
            embedder.model.eval()
        patch_table.obsm[embedding_name] = _embed_patches_vision(
            sdata, embedder,
            image_name=image_name,
            point_name=point_name,
            patch_name=patch_name,
            patch_table=patch_table,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            precache=precache,
            color_transform=color_transform,
            block_px=block_px,
            embedding_dtype=embedding_dtype,
            use_amp=use_amp,
            pin_memory=pin_memory,
        )
    elif model_type == "SAE":
        sae_embedding = embedder.transform(patch_table.obsm[obsm_key], device=device, 
                                           column_keep_indices=column_keep_indices)
        if column_keep_indices is not None:
            new_vars = [f'{embedder_name}_{i}' for i in column_keep_indices]
        else:
            new_vars = [f'{embedder_name}_{i}' for i in range(sae_embedding.shape[1])]
        existing_vars = list(patch_table.var.index)
        # slice columns of the table to remove these columns that are redundant with embedding
        diff_vars = [item for item in existing_vars if item not in new_vars]
        if len(diff_vars) == 0 and (not overwrite):
            return sdata
        if len(diff_vars) < len(existing_vars) or overwrite:
            patch_table = patch_table[:, diff_vars].copy() # may be slow with large table?
        patch_table.var.index = pd.Index(diff_vars + new_vars)
        patch_table.X = hstack((csc_array(patch_table.X), sae_embedding.tocsc())).tocsr()
        sdata[patch_name_full] = patch_table
    elif model_type == "KMeans":
        base_embeddings = patch_table.obsm[obsm_key]
        kmeans_labels = embedder.transform(base_embeddings)
        label_col_name = f'{embedder_name}_label'
        patch_table.obs[label_col_name] = kmeans_labels
        for i in range(embedder.n_clusters):
            patch_table.obs[f'{embedder_name}_label_{i}'] = (kmeans_labels == i).astype(int)
        
    if save:
        overwrite_element(sdata, patch_name_full)
    return sdata