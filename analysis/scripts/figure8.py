"""Make Figure 8 for paper.

"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import sigpy as sp

from os import path, makedirs

from toniq.lattice import make_lattice
from toniq.linop import get_matrix
from toniq.sr import forward_model
from toniq.plot_params import *

def get_condition(kspace, psf_shape, lamda=0):
    A_op = forward_model(kspace, psf_shape)
    # A_op = forward_model_conv(kspace, psf_shape)
    # print(A_op)
    A_mtx = get_matrix(A_op, verify=True)
    # print(A_mtx.shape)
    if lamda != 0:
        A_mtx = np.vstack((A_mtx, np.eye(A_mtx.shape[-1]) * np.sqrt(lamda)))
    return np.linalg.cond(A_mtx)

def plot_condition(ax, psf_sizes, cubic, gyroid):
    ax.scatter(psf_sizes, cubic, label=r'Cubic Lattice', marker='s')
    ax.scatter(psf_sizes, gyroid, label=r'Gyroid Lattice', marker='.')
    ax.set_xlabel('PSF Kernel Size (pixels)')
    ax.set_ylabel('Condition Number')
    ax.legend()
    # plt.yscale('log')
    # ax.set_xlim([min(psf_sizes), max(psf_sizes)])
    ax.set_ylim([0, 100])
    # plt.grid()
    return ax

def get_kspace_center(lattice, init_shape, final_shape=None):
    kspace = sp.ifft(sp.resize(sp.fft(lattice), init_shape), axes=(2,))
    if final_shape is not None:
        kspace = sp.fft(sp.resize(sp.ifft(kspace, axes=(0, 1)), final_shape), axes=(0, 1))
    # kspace = sp.ifft(sp.resize(sp.fft(lattice), shape), axes=(0, 1, 2)) # temp fix for conv model
    return kspace

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
    plt.savefig(path.join(args.save_dir, 'figure8.pdf'), dpi=DPI)

    if args.plot:
        plt.show()