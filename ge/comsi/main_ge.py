import argparse
from glob import glob
import json
import numpy as np
from os import path

from cfl import readcfl
from ge_scanarchive import ScanArchive
from util import debug

p = argparse.ArgumentParser(description='extract MAVRIC-SL scans from GE Scan Archives; '
                                        'create seq.yml, data.npy, mask.npy for each scan')
p.add_argument('root', type=str, help='path to scan data folders containing h5 files in Scan Archive format')
p.add_argument('-s', '--skip', action='store_true', help='skip extraction')
p.add_argument('-m', '--matlab', action='store_true', help='use MATLAB version of orchestra')
p.add_argument('-n', '--series_num', type=int, default=None, help='if specified, extract just this one series')


def largest_file(d, ext):
    """ Find the largest file with extension ext in directory d. """
    files = glob(path.join(d, '*.{}'.format(ext)))
    return sorted((path.getsize(f), f) for f in files)[-1][1]


def reduce(k, mask, bin_slc=slice(None), coil_slc=slice(None)):
    mask = mask[..., bin_slc]
    k = k[..., bin_slc, coil_slc]
    return k, mask


if __name__ == '__main__':

    args = p.parse_args()
    with open(path.join(args.root, 'args_sim.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    root = path.abspath(args.root)
    flipread = True
    for scan_dir in glob(f'{root}/Series*'):
        if args.series_num is not None and path.split(scan_dir)[-1] != 'Series{}'.format(args.series_num):
            print('Skipping {}'.format(scan_dir))
            continue
        if not args.skip:
            t0 = debug('Extracting {}{}'.format(scan_dir, ', flipping readout' if flipread else ''))
            file = largest_file(scan_dir, 'h5')
            with open(path.join(scan_dir, 'path_to_scan_archive'), 'w') as f:
                f.write(file)
            if args.matlab:
                import ge_matlab as gemat
                s = gemat.ScanArchive(file, verbose=True)
                s.extract_params()
                s.extract_data(flipread=flipread)
                # IF YOU GET "SystemError: MATLAB process cannot be terminated." THEN THE MATLAB SCRIPT HAS FAILED.
                # MOST LIKELY DUE TO INSUFFICIENT MEMORY. EXPECT 32+ GB of RAM FOR FULLY SAMPLED DATA.
            else:
                s = ScanArchive(file, verbose=True)
                s.save_params(path.join(scan_dir, 'seq.yml'))
                k, mask, control_table = s.extract_data(flipread=flipread)
            # flipread = not flipread
            debug('done extraction', t0)
        if args.matlab:
            mask = readcfl(path.join(scan_dir, 'mask_yzb')).astype(bool)
            k = readcfl(path.join(scan_dir, 'kspace_kbc'))
        k, mask = reduce(k, mask)
        mask = np.broadcast_to(mask, (s.nx, s.ny, s.nz, s.nb))
        np.save(path.join(scan_dir, 'data.npy'), k)  # (k, b, c)
        np.save(path.join(scan_dir, 'mask.npy'), mask)  # (x, y, z, b)
