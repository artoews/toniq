import numpy as np
import matplotlib.pyplot as plt
from os import path
import sigpy as sp

from plot import plotVolumes
from plot_lattice import plot_cells
from lattice import make_lattice
from plot_params import *
from slice_params import *
from util import equalize, load_series

exam_root = '/Users/artoews/root/data/mri/240202/14446_dicom/'
series = 'Series4' # 512x512
save_dir = '/Users/artoews/root/code/projects/metal-phantom/feb2'

lattice_slc = (slice(None),) + tuple(slice(s.start*2, s.stop*2) for s in LATTICE_SLC[:2]) + (LATTICE_SLC[2],)
slc = (slice(None), slice(None), 18)
inset_shape = (40, 40, 20)
inset = (slice(160, 200), slice(160, 200))
crop_shape = (256, 172)
psf_shape = (9, 9)

def plot_panel(ax, image, kspace=False):
    if kspace:
        vmax = 2
        image = np.abs(sp.fft(image))
        # image = np.log(image + 1)
    else:
        vmax = 1
    ax.imshow(image, cmap=CMAP['image'], vmin=0, vmax=vmax) 
    ax.axis('off')

def make_lattice_to_shape(cell, shape):
    lattice = make_lattice(cell, resolution=1, shape=(2, 2, 2))
    lattice = np.abs(sp.ifft(sp.resize(sp.fft(lattice), shape)))
    lattice = lattice / np.max(lattice) / 1.5
    return lattice

def plot_lattice(fig, cube, gyroid):
    num_slices = 5
    axes = fig.subplots(nrows=4, ncols=num_slices)
    for i in range(num_slices):
        plot_panel(axes[0, i], cube[:, :, i])
        plot_panel(axes[1, i], cube[:, :, i], kspace=True)
        plot_panel(axes[2, i], gyroid[:, :, i])
        plot_panel(axes[3, i], gyroid[:, :, i], kspace=True)
    return axes

def plot_model(fig, reference, target, inset, psf):
    axes = fig.subplots(nrows=3, ncols=3)
    plot_panel(axes[0, 0], target)
    plot_panel(axes[1, 0], target[inset])
    plot_panel(axes[2, 0], target[inset], kspace=True)
    plot_panel(axes[0, 1], reference)
    plot_panel(axes[1, 1], reference[inset])
    plot_panel(axes[2, 1], reference[inset], kspace=True)
    axes[0, 2].axis('off')
    plot_panel(axes[1, 2], psf)
    plot_panel(axes[2, 2], psf, kspace=True)
    return axes

def downsample(image, shape):
    k = sp.fft(image)
    k = sp.resize(sp.resize(k, shape), image.shape)
    return np.abs(sp.ifft(k))

def make_psf(init_shape, retro_shape, psf_shape):
    psf = sp.ifft(np.ones(init_shape))
    psf = downsample(psf, retro_shape)
    psf = psf / np.max(np.abs(psf))
    psf = sp.resize(psf, psf_shape)
    return psf

image = load_series(exam_root, series).data
crop_shape = crop_shape + (image.shape[-1],)
images = [image, downsample(image, crop_shape)]
psf = make_psf(image.shape[:2], crop_shape, inset_shape[:2])
images = equalize(np.stack(images))[lattice_slc]

cubic_lattice = make_lattice_to_shape('cubic', inset_shape)
gyroid_lattice = make_lattice_to_shape('gyroid', inset_shape)

# fig, tracker = plotVolumes((images[0], images[1]))

# fig = plt.figure(figsize=(12, 6), layout='constrained')
# fig_A, fig_B, fig_C = fig.subfigures(1, 3, wspace=0.1, width_ratios=[2, 1, 1])

save_args = {'dpi': 300, 'bbox_inches': 'tight', 'pad_inches': 0}

fig_A = plt.figure(figsize=(10, 8), layout='constrained')
plot_model(fig_A, images[0][slc], images[1][slc], inset, psf)
plt.savefig(path.join(save_dir, 'figure4A.png'), **save_args)

fig_B = plt.figure(figsize=(10, 8), layout='constrained')
plot_lattice(fig_B, cubic_lattice, gyroid_lattice)
plt.savefig(path.join(save_dir, 'figure4B.png'), **save_args)

plt.show()
quit()

fig_C = plt.figure(figsize=(4, 8), layout='constrained')
cubic_lattice = make_lattice('cubic', resolution=1, shape=(2, 2, 2))
cubic_lattice = np.roll(np.roll(cubic_lattice, -1, axis=0), -1, axis=2) # better form for visualization purposes
gyroid_lattice = make_lattice('gyroid', resolution=1, shape=(2, 2, 2))
plot_cells(fig_C, cubic_lattice, gyroid_lattice)
plt.savefig(path.join(save_dir, 'figure4C.png'), **save_args)

plt.show()
