"""Make Figure 5 for paper.

"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from os import path, makedirs
import sigpy as sp

from toniq.plot import remove_ticks, color_panels, label_panels, plotVolumes
from toniq.lattice import make_lattice
from toniq.plot_params import *

kwargs = {'vmin': 0, 'vmax': 1.2, 'cmap': CMAP['image']}

def plot_cell(ax, vol, vmax=1):
    filled = np.ones_like(vol, dtype=bool)
    filled[1:-1, 1:-1, 1:-1] = False
    facecolors = np.empty(vol.shape + (3,), dtype=np.float64)
    facecolors[vol==0] = np.zeros(3)
    facecolors[vol==1] = np.ones(3) / vmax
    ax.voxels(filled, facecolors=facecolors, edgecolors=facecolors)
    ax.set(xlabel='x', ylabel='y', zlabel='z')
    ax.set_aspect('equal')
    ax.set_axis_off()
    ax.patch.set_alpha(0)
    return ax

def make_lattice_to_shape(cell, shape):
    lattice = make_lattice(cell, resolution=1, shape=(1, 1, 1))
    lattice = np.abs(sp.ifft(sp.resize(sp.fft(lattice), shape)))
    lattice = lattice / np.max(lattice)
    return lattice

def plot_slices(fig, lattice):
    num_slices = 5
    axes = fig.subplots(nrows=2, ncols=num_slices, gridspec_kw={'wspace': 0, 'hspace': 0, 'left': 0.05, 'right': 0.95, 'bottom': 0.05, 'top': 0.95})
    for i in range(num_slices):
        image = lattice[:, :, i]
        kspace = np.abs(sp.fft(image))
        kspace = np.log(kspace + 1)
        axes[0, i].imshow(image, **kwargs)
        axes[1, i].imshow(kspace, **kwargs)
    axes[0, 0].set_ylabel('Image')
    axes[1, 0].set_ylabel('2D DFT')
    remove_ticks(axes)
    return axes

def plot_lattice(subfig, lattice):
    ax = subfig.subplots(subplot_kw={'projection': '3d'}, gridspec_kw={'left': 0, 'right': 1, 'bottom': 0, 'top': 1})
    plot_cell(ax, lattice, vmax=kwargs['vmax'])

p = argparse.ArgumentParser(description='Make figure 5')
p.add_argument('save_dir', type=str, help='path where figure is saved')
p.add_argument('--inset_size', type=int, default=10, help='path where figure is saved')
p.add_argument('-p', '--plot', action='store_true', help='show plots')

if __name__ == '__main__':

    args = p.parse_args()
    if not path.exists(args.save_dir):
        makedirs(args.save_dir)

    fig = plt.figure(figsize=(FIG_WIDTH[2], FIG_WIDTH[2]*0.55))
    subfigs = fig.subfigures(2, 2, wspace=0.03, hspace=0.05, width_ratios=[1, 2.5])

    gyroid_lattice = make_lattice('gyroid', resolution=1, shape=(1, 1, 1))
    cubic_lattice = make_lattice('cubic', resolution=1, shape=(1, 1, 1))
    # fig, tracker = plotVolumes((cubic_lattice, gyroid_lattice))
    plot_lattice(subfigs[0, 0], gyroid_lattice)
    plot_lattice(subfigs[1, 0], cubic_lattice)

    gyroid_lattice = make_lattice_to_shape('gyroid', (args.inset_size,) * 3)
    cubic_lattice = make_lattice_to_shape('cubic', (args.inset_size,) * 3)
    plot_slices(subfigs[0, 1], gyroid_lattice)
    plot_slices(subfigs[1, 1], cubic_lattice)

    color_panels(subfigs.flat)
    label_panels(subfigs.flat)

    plt.savefig(path.join(args.save_dir, 'figure5.png'), dpi=DPI)
    plt.savefig(path.join(args.save_dir, 'figure5.pdf'), dpi=DPI)

    if args.plot:
        plt.show()