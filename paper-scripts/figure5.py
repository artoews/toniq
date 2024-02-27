import numpy as np
import matplotlib.pyplot as plt
from os import path
import sigpy as sp

from plot import remove_ticks, color_panels, label_panels
from plot_lattice import plot_cell
from lattice import make_lattice
from plot_params import *
from slice_params import *

root = '/Users/artoews/root/code/projects/metal-phantom/feb2/'
inset_shape = (10, 10, 10)
kwargs = {'vmin': 0, 'vmax': 1, 'cmap': CMAP['image']}

def make_lattice_to_shape(cell, shape):
    lattice = make_lattice(cell, resolution=1, shape=(1, 1, 1))
    lattice = np.abs(sp.ifft(sp.resize(sp.fft(lattice), shape)))
    lattice = lattice / np.max(lattice) / 1.5
    return lattice

def plot_slices(fig, lattice):
    num_slices = 5
    axes = fig.subplots(nrows=2, ncols=num_slices, gridspec_kw={'wspace': 0, 'hspace': 0, 'left': 0.05, 'right': 0.95, 'bottom': 0.05, 'top': 0.95})
    for i in range(num_slices):
        image = lattice[:, :, i]
        kspace = np.abs(sp.fft(image))
        kspace = np.log(kspace + 1) * 2
        axes[0, i].imshow(image, **kwargs)
        axes[1, i].imshow(kspace, **kwargs)
    axes[0, 0].set_ylabel('Image')
    axes[1, 0].set_ylabel('K-space')
    remove_ticks(axes)
    return axes

def plot_lattice(subfig, lattice):
    ax = subfig.subplots(subplot_kw={'projection': '3d'}, gridspec_kw={'left': 0, 'right': 1, 'bottom': 0, 'top': 1})
    plot_cell(ax, lattice)

fig = plt.figure(figsize=(FIG_WIDTH[2], FIG_WIDTH[2]*0.55))
subfigs = fig.subfigures(2, 2, wspace=0.03, hspace=0.05, width_ratios=[1, 2.5])

cubic_lattice = make_lattice('cubic', resolution=1, shape=(1, 1, 1))
cubic_lattice = np.roll(np.roll(cubic_lattice, -1, axis=0), -1, axis=2) # better form for visualization purposes
gyroid_lattice = make_lattice('gyroid', resolution=1, shape=(1, 1, 1))
plot_lattice(subfigs[0, 0], cubic_lattice)
plot_lattice(subfigs[1, 0], gyroid_lattice)

cubic_lattice = make_lattice_to_shape('cubic', inset_shape)
gyroid_lattice = make_lattice_to_shape('gyroid', inset_shape)
plot_slices(subfigs[0, 1], cubic_lattice)
plot_slices(subfigs[1, 1], gyroid_lattice)

color_panels(subfigs.flat)
label_panels(subfigs.flat)

plt.savefig(path.join(root, 'figure5.png'), dpi=DPI)

plt.show()