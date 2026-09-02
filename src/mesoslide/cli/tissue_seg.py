import argparse
import shutil
import zarr
import numpy as np
import cv2
from numcodecs import Zstd
from skimage.filters.rank import entropy
from skimage.filters import threshold_otsu
from skimage.measure import label
from skimage.morphology import remove_small_objects
from meson._tifffile_wsi import TiffFile
from meson.preprocessing._normalize_mIF import channelwise_tissue_segmentation

tissue_segmentation = channelwise_tissue_segmentation

def finalize_to_zip(out_path):
    zip_path = out_path.rstrip('/').rstrip('.zarr') + '.zarr.zip'
    store = zarr.DirectoryStore(out_path)
    with zarr.ZipStore(zip_path, mode='w') as zip_store:
        zarr.copy_store(store, zip_store)
    shutil.rmtree(out_path)
    print(f"Written to {zip_path}, removed {out_path}")

def init_store(wsi_path, out_path, nchan_out):
    wsi_handle = TiffFile(wsi_path)
    C, H, W = wsi_handle[0][0].shape

    store = zarr.DirectoryStore(out_path)
    root = zarr.group(store=store, overwrite=False)
    if '0' not in root:
        root.zeros(
            name='0',
            shape=(C, H, W),
            chunks=(1, 1024, 1024),
            dtype=bool,
            compressor=Zstd(level=3)
        )

    with open(nchan_out, 'w') as f:
        f.write(str(C))
    print(f"C={C}, array initialized at {out_path}")

def main():
    parser = argparse.ArgumentParser(prog='tissue-seg')
    parser.add_argument('--wsi_path', required=True)
    subparsers = parser.add_subparsers(dest='mode', required=True)

    seg_parser = subparsers.add_parser('segment')
    seg_parser.add_argument('--channel', type=int, required=True)
    seg_parser.add_argument('--out_path', required=True)
    seg_parser.add_argument('--mpp', type=float, required=True)
    
    finalize_parser = subparsers.add_parser('finalize')
    finalize_parser.add_argument('--out_path', required=True)
    
    init_parser = subparsers.add_parser('init')
    init_parser.add_argument('--out_path', required=True)
    init_parser.add_argument('--nchan_out', required=True)
    
    args = parser.parse_args()

    if args.mode == 'segment':
        tissue_segmentation(args.wsi_path, args.out_path, args.channel, args.mpp)
    elif args.mode == 'finalize':
        finalize_to_zip(args.out_path)
    elif args.mode == 'init':
        init_store(args.wsi_path, args.out_path, args.nchan_out)


if __name__ == '__main__':
    main()