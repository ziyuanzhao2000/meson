import numpy as np

def get_optimal_chunk_size(image, patch_chunk_size=256, target_size=500*1024*1024):
    n_channels = len(image['c'])
    element_size = np.dtype(image.dtype).itemsize
    patch_size = element_size * patch_chunk_size * patch_chunk_size * n_channels
    num_patches = max(1, int(target_size/patch_size))
    chunk_size = patch_chunk_size * int(num_patches**0.5)
    print("new chunk size", [n_channels, chunk_size, chunk_size])
    return [n_channels, chunk_size, chunk_size]