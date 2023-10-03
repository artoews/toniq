import argparse
import json
import numpy as np
from os import path, makedirs
from pathlib import Path

import dicom

p = argparse.ArgumentParser(description='Image quality mapping toolbox for image volumes from metal phantom scans')
p.add_argument('out', type=str, help='path where outputs are saved')
p.add_argument('target_image', type=str, help='path to image volume; target for analysis')
p.add_argument('-c', '--clean_image', type=str, default=None, help='path to image volume; reference for analysis; default=None')
p.add_argument('-d', '--repeat_image', type=str, default=None, help='path to image volume; repetition of target; default=None')
p.add_argument('-s', '--snr', action='store_true', help='map SNR, and nothing else unless explicitly indicated')
p.add_argument('-r', '--resolution', action='store_true', help='map resolution, and nothing else unless explicitly indicated')
p.add_argument('-g', '--geometric', action='store_true', help='map geometric distortion, and nothing else unless explicitly indicated')
p.add_argument('-i', '--intensity', action='store_true', help='map intensity distortion, and nothing else unless explicitly indicated')

p.add_argument('-n', '--num_workers', type=int, default=8, help='number of workers used for parallelized tasks (mapping resolution); default=8')

p.add_argument('-v', '--verbose', action='count', default=0, help='verbosity level')


def setup_dirs(root):
    map_dir = path.join(root, 'map')
    plot_dir = path.join(root, 'plot')
    dirs = (map_dir, plot_dir)
    for d in dirs:
        if not path.exists(d):
            makedirs(d)
    return dirs 

def load_dicom_series(path):
    files = Path(path).glob('*MRDC*')
    image = dicom.load_series(files)
    return image


if __name__ == '__main__':
    args = p.parse_args()
    with open(path.join(args.out, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    map_dir, plot_dir = setup_dirs(args.out)

    target_image = load_dicom_series(args.target_image)

    print('Target image', target_image.shape)


