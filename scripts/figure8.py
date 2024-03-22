import argparse
import numpy as np
import matplotlib.pyplot as plt

from os import path, makedirs

from lattice import make_lattice, get_kspace_center, get_condition

from plot_params import *

def plot_condition(ax, psf_sizes, cubic, gyroid):
    ax.scatter(psf_sizes, cubic, label=r'Cubic Lattice', marker='s')
    ax.scatter(psf_sizes, gyroid, label=r'Gyroid Lattice', marker='.')
    ax.set_xlabel('PSF Size (pixels)')
    ax.set_ylabel('Condition Number')
    ax.legend()
    # plt.yscale('log')
    # ax.set_xlim([min(psf_sizes), max(psf_sizes)])
    ax.set_ylim([0, 100])
    # plt.grid()
    return ax

p = argparse.ArgumentParser(description='Make figure 8')
p.add_argument('save_dir', type=str, help='path where figure is saved')
p.add_argument('-p', '--plot', action='store_true', help='show plots')

if __name__ == '__main__':

    args = p.parse_args()
    if not path.exists(args.save_dir):
        makedirs(args.save_dir)

    lattice_res = 1
    # for the 10x10x10 to approximate implementation, need to suppress final crop in the forward model; but the 20x20x10->14x14x10 case below is more accurate anyway
    # init_patch_shape = (10, 10, 10) # for cropping in k-space
    # cell_shape = (1, 1, 1)
    # final_patch_shape = None
    init_patch_shape = (20, 20, 10) # for cropping in k-space
    cell_shape = (2, 2, 1)
    final_patch_shape = (14, 14, 10) # for cropping in image space

    psf_sizes = range(1, 9)
    psf_shapes = [(size, size, 1) for size in psf_sizes]
    
    gyroid = make_lattice('gyroid', resolution=lattice_res, shape=cell_shape)
    gyroid_k = get_kspace_center(gyroid, init_patch_shape, final_shape=final_patch_shape)
    cubic = make_lattice('cubic', resolution=lattice_res, shape=cell_shape)
    cubic_k = get_kspace_center(cubic, init_patch_shape, final_shape=final_patch_shape)

    condition_cubic = [get_condition(cubic_k, shape, lamda=0) for shape in psf_shapes]
    condition_gyroid = [get_condition(gyroid_k, shape, lamda=0) for shape in psf_shapes]
    # print('cubic condition', condition_cubic)
    # print('gyroid condition', condition_gyroid)
    
    fig, axes = plt.subplots(figsize=(FIG_WIDTH[0], FIG_WIDTH[0]), layout='constrained')
    plot_condition(axes, psf_sizes, condition_cubic, condition_gyroid)

    plt.savefig(path.join(args.save_dir, 'figure8.png'), dpi=DPI)

    if args.plot:
        plt.show()