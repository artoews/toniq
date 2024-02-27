import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import sigpy as sp

from os import path

from slice_params import *
from plot_params import *
from plot import remove_ticks, color_panels, label_panels

kwargs = {'vmin': 0, 'vmax': 1, 'cmap': CMAP['image']}

def plot_inputs(fig, target, reference, inset):
    axes = fig.subplots(2, 1, gridspec_kw={'left': 0.05, 'right': 0.95, 'bottom': 0.05, 'top': 0.92})
    axes[0].imshow(reference, **kwargs)
    axes[1].imshow(target, **kwargs)
    axes[0].set_title('Reference')
    axes[1].set_title('Target')
    remove_ticks(axes)
    inset_start = (inset[0].start, inset[1].start)
    inset_size = (inset[0].stop - inset[0].start, inset[1].stop - inset[1].start)
    for ax in axes.flat:
        rect = patches.Rectangle(inset_start, *inset_size, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

def plot_model(fig, target, reference, psf):
    axes = fig.subplots(2, 3, gridspec_kw={'left': 0.05, 'right': 0.95, 'bottom': 0.05, 'top': 0.92})
    titles = ('Target', 'Reference', 'Local PSF')
    images = (target, reference, psf)
    for i in range(3):
        axes[0, i].imshow(images[i], **kwargs)
        axes[0, i].set_title(titles[i])
        kspace = np.abs(sp.fft(images[i]))
        kspace = np.log(kspace + 1) * 2
        axes[1, i].imshow(kspace, **kwargs)
    axes[0, 0].set_ylabel('Patch')
    axes[1, 0].set_ylabel('K-space')
    symbols = ('=', r'$\circledast$', '=', r'$\odot$')
    for ax, symbol in zip(axes[:, :2].flat, symbols):
        ax.text(1.1, 0.49, symbol, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, size=20)
    remove_ticks(axes)

root = '/Users/artoews/root/code/projects/metal-phantom/feb2/'
slc = (slice(None), slice(None), 18)
inset = (slice(87, 96), slice(87, 96))
psf_shape = (5, 5)

images = np.load(path.join(root, 'resolution', 'images.npy'))
reference = images[0][slc]
target = images[1][slc]

fig = plt.figure(figsize=(FIG_WIDTH[2], FIG_WIDTH[2]*0.5))
subfigs = fig.subfigures(1, 2, width_ratios=[1, 3], wspace=0.04)

plot_inputs(subfigs[0], target, reference, inset)
plot_model(subfigs[1], target[inset], reference[inset], reference[inset])

label_panels(subfigs)
color_panels(subfigs)

plt.savefig(path.join(root, 'figure4.png'), dpi=DPI)

plt.show()