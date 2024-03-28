import argparse
import numpy as np

from os import path, makedirs

from config import read_config, load_volume
from plot_params import *

import gd
from config import read_config
from plot import plotVolumes

from util import equalize

p = argparse.ArgumentParser(description='Check for positional drift of phantom')
p.add_argument('save_dir', type=str, help='path where figure is saved')
p.add_argument('-c', '--config', type=str, default='config/mar4-fse125.yml', help='yaml config file for sequence')
# p.add_argument('-z', '--z_slice', type=int, default=50, help='z index of slice')
p.add_argument('-p', '--plot', action='store_true', help='show plots')

if __name__ == '__main__':

    args = p.parse_args()
    if not path.exists(args.save_dir):
        makedirs(args.save_dir)

    config = read_config(args.config)

    image1 = load_volume(config, 'first-scan').data
    image2 = load_volume(config, 'last-scan').data
    image1, image2 = equalize([image1, image2])

    slc = (slice(None), slice(None), slice(None))
    # slc = (slice(40, 70), slice(100, 150), slice(12, 24)) # bottom half
    slc = (slice(40, 70), slice(100, 150), slice(35, 47)) # top half
    image1 = image1[slc]
    image2 = image2[slc]

    result, rigid_transform = gd.elastix_registration(image1, image2, None, None, gd.setup_rigid(), verbose=False)
    print(rigid_transform) # with order 3D rotation (rad?), 3D translation (pixels?)

    fig, tracker = plotVolumes((image1, image2, 2*np.abs(image2 - image1), result, 2*np.abs(result - image1)))

    if args.plot:
        plt.show()
