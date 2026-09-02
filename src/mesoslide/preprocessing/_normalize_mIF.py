import os
import shutil
import gc
import numexpr as ne
from tqdm import tqdm
import zarr
import numpy as np
import cv2
import tifffile
from numcodecs import Zstd
from skimage.filters.rank import entropy
from skimage.filters import threshold_otsu
from skimage.measure import label
from skimage.morphology import remove_small_objects
from scipy.ndimage import binary_dilation
from meson._tifffile_wsi import TiffFile

def channelwise_tissue_segmentation(wsi_path, out_path, channel_idx, mpp):
    wsi_handle = TiffFile(wsi_path)
    shape = wsi_handle[0][0].shape
    if len(shape) == 3:
        C, H, W = shape
        flat = False
    else:
        H, W = shape
        C = 1
        flat = True
    store = zarr.DirectoryStore(out_path)
    root = zarr.group(store=store, overwrite=False)
    compressor = Zstd(level=3)
    
    if '0' in root and isinstance(root['0'], zarr.core.Array):
        arr = root['0']
    else:
        if flat:
            arr = root.zeros(
                name='0', 
                shape=(H, W), 
                chunks=(1024, 1024), 
                dtype=np.bool, 
                compressor=Zstd(level=3)
            )
        else:
            arr = root.zeros(
                name='0', 
                shape=(C, H, W), 
                chunks=(1, 1024, 1024), 
                dtype=np.bool, 
                compressor=Zstd(level=3)
            )
    
    num_pixels = 5 / mpp
    level = int(np.ceil(np.log2(num_pixels)))
    if flat:
        chn_thumbnail = np.log(wsi_handle[0][level].data.original[:] + 1)
    else:
        chn_thumbnail = np.log(wsi_handle[0][level].data.original[channel_idx] + 1)
    chn_entropy = entropy((chn_thumbnail - chn_thumbnail.mean()) / (chn_thumbnail.max() - chn_thumbnail.min()), 
                          np.ones((5, 5)))
    thresh = threshold_otsu(chn_entropy.flatten())
    mask = chn_entropy > thresh
    mask = binary_dilation(mask, np.ones((26, 26)))
    mask = remove_small_objects(mask, min_size=40000) > 0 # 1 mm^2
    mask_resized = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
    if flat:
        arr[:, :] = mask_resized
    else:
        arr[channel_idx, :, :] = mask_resized


# def compute_interpolated_percentiles(image: np.ndarray, L: int, low_pct: float, high_pct: float):
#     """
#     Computes local non-overlapping L x L percentiles and bilinearly 
#     interpolates them to the original image shape.
    
#     Args:
#         image: 2D numpy array of shape (H, W), typically np.int16
#         L: integer, tile size
#         low_pct: float, lower percentile (e.g., 5.0)
#         high_pct: float, upper percentile (e.g., 95.0)
        
#     Returns:
#         low_map, high_map: Interpolated arrays of shape (H, W) as np.float32
#     """
#     h, w = image.shape
    
#     pad_h = (L - h % L) % L
#     pad_w = (L - w % L) % L
    
#     img_padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect')
#     H_pad, W_pad = img_padded.shape
    
#     grid_h, grid_w = H_pad // L, W_pad // L
    
#     tiles = img_padded.reshape(grid_h, L, grid_w, L)
#     tiles = tiles.swapaxes(1, 2)
#     tiles = tiles.reshape(grid_h, grid_w, L * L)
    
#     k_low  = max(0, int(np.floor(low_pct  / 100 * (L*L - 1))))
#     k_high = max(0, int(np.floor(high_pct / 100 * (L*L - 1))))

#     low_grid  = np.partition(tiles, k_low,  axis=-1)[..., k_low ].astype(np.float32)
#     high_grid = np.partition(tiles, k_high, axis=-1)[..., k_high].astype(np.float32)
    
#     low_interp = cv2.resize(low_grid, (W_pad, H_pad), interpolation=cv2.INTER_LINEAR)
#     high_interp = cv2.resize(high_grid, (W_pad, H_pad), interpolation=cv2.INTER_LINEAR)
    
#     low_final = low_interp[:h, :w]
#     high_final = high_interp[:h, :w]
    
#     return low_final, high_final

def compute_interpolated_percentiles(image: np.ndarray, L: int, low_pct: float, high_pct: float):
    h, w = image.shape
    
    pad_h = (L - h % L) % L
    pad_w = (L - w % L) % L
    
    img_padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect')
    H_pad, W_pad = img_padded.shape
    
    grid_h, grid_w = H_pad // L, W_pad // L
    
    low_grid = np.zeros((grid_h, grid_w), dtype=np.float32)
    high_grid = np.zeros((grid_h, grid_w), dtype=np.float32)
    
    for i in tqdm(range(grid_h)):
        for j in range(grid_w):
            block = img_padded[i*L:(i+1)*L, j*L:(j+1)*L].flatten()
            low_grid[i, j] = np.percentile(block, low_pct)
            high_grid[i, j] = np.percentile(block, high_pct)
            
    del img_padded 
    
    low_interp = cv2.resize(low_grid, (W_pad, H_pad), interpolation=cv2.INTER_LINEAR)
    high_interp = cv2.resize(high_grid, (W_pad, H_pad), interpolation=cv2.INTER_LINEAR)
    
    return low_interp[:h, :w], high_interp[:h, :w]

def channelwise_normalization(wsi_path, mask_path, out_path, channel_idx, mpp):
    level = int(np.ceil(np.log2(1 / mpp)))
    pixel_spacing = 2**level
    wsi_handle = TiffFile(wsi_path)
    
    shape = wsi_handle[0][0].shape
    if len(shape) == 3:
        flat = False
        C, H, W = shape
    else:
        flat = True
        H, W = shape
        C = 1
    store = zarr.DirectoryStore(out_path)
    root = zarr.group(store=store, overwrite=False)
    compressor = Zstd(level=3)
    
    if '0' in root and isinstance(root['0'], zarr.core.Array):
        arr = root['0']
    else:
        if flat:
            arr = root.zeros(
                name='0', 
                shape=(H, W), 
                chunks=(1024, 1024), 
                dtype=np.uint8, 
                compressor=Zstd(level=3)
            )
        else:
            arr = root.zeros(
                name='0', 
                shape=(C, H, W), 
                chunks=(1, 1024, 1024), 
                dtype=np.uint8, 
                compressor=Zstd(level=3)
            )
            
    if flat:
        raw_chn = wsi_handle[0][0].data.original[:]
    else:        
        raw_chn = wsi_handle[0][0].data.original[channel_idx]
    print(level, raw_chn.dtype, raw_chn.shape)
    chn = raw_chn.astype(np.float32)
    chn_ds = chn[::pixel_spacing, ::pixel_spacing]
    np.log1p(chn, out=chn) 
    del raw_chn
    
    tile_size = int(128 / (mpp * pixel_spacing))
    print(tile_size)
    low_, high_ = compute_interpolated_percentiles(chn_ds, tile_size, low_pct=5, high_pct=99.9)
    del chn_ds
    print("Computed percentile maps")
    high_resized = cv2.resize(high_, (W, H), interpolation=cv2.INTER_LINEAR)
    low_resized = cv2.resize(low_, (W, H), interpolation=cv2.INTER_LINEAR)
    del low_, high_
    print("Resized percentile maps")
    # chn_normalized = (chn - low_resized)
    # chn_normalized /= (high_resized - low_resized + 1e-5)
    # np.clip(chn_normalized, 0, 1, out=chn_normalized)
    # np.multiply(chn_normalized, mask * 255, out=chn_normalized)
    with zarr.ZipStore(mask_path, mode='r') as mask_store:
        mask_handle = zarr.group(store=mask_store, overwrite=False)['0']
        if flat:
            mask = mask_handle[:].astype(bool)
        else:
            mask = mask_handle[channel_idx].astype(bool)
        
    chn_normalized = ne.evaluate(
        "where((chn - low_resized) / (high_resized - low_resized + 1e-5) < 0, 0, "
        "where((chn - low_resized) / (high_resized - low_resized + 1e-5) > 1, 1, "
        "(chn - low_resized) / (high_resized - low_resized + 1e-5))) * mask * 255"
    )
    del chn, low_resized, high_resized, mask
    chn_normalized = chn_normalized.astype(np.uint8)

    chn_normalized = chn_normalized.astype(np.uint8)
    print("Normalized channel")
    if flat:
        arr[:, :] = chn_normalized
    else:   
        arr[channel_idx, :, :] = chn_normalized