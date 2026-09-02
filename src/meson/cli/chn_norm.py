import argparse
import shutil
import zarr
import numpy as np
from numcodecs import Zstd
from meson._tifffile_wsi import TiffFile
from meson.preprocessing._normalize_mIF import channelwise_normalization


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
            dtype=np.uint8,          # uint8, not bool
            compressor=Zstd(level=3)
        )

    with open(nchan_out, 'w') as f:
        f.write(str(C))
    print(f"C={C}, array initialized at {out_path}")


def main():
    parser = argparse.ArgumentParser(prog='chn-norm')
    parser.add_argument('--wsi_path', required=True)
    parser.add_argument('--mask_path', required=True)   # extra arg vs tissue-seg
    subparsers = parser.add_subparsers(dest='mode', required=True)

    norm_parser = subparsers.add_parser('normalize')
    norm_parser.add_argument('--channel', type=int, required=True)
    norm_parser.add_argument('--out_path', required=True)
    norm_parser.add_argument('--mpp', type=float, required=True)

    finalize_parser = subparsers.add_parser('finalize')
    finalize_parser.add_argument('--out_path', required=True)

    init_parser = subparsers.add_parser('init')
    init_parser.add_argument('--out_path', required=True)
    init_parser.add_argument('--nchan_out', required=True)

    args = parser.parse_args()

    if args.mode == 'normalize':
        channelwise_normalization(args.wsi_path, args.mask_path, args.out_path, args.channel, args.mpp)
    elif args.mode == 'finalize':
        finalize_to_zip(args.out_path)
    elif args.mode == 'init':
        init_store(args.wsi_path, args.out_path, args.nchan_out)


if __name__ == '__main__':
    main()