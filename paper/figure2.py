import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.transforms as mtransforms

from os import path

from plot_artifact import plot_ia_map
from plot_distortion import plot_gd_map
from plot_snr import plot_snr_map
from plot_resolution import plot_res_map
from plot_params import *
from plot import remove_ticks, color_panels, label_panels
from string import ascii_uppercase

def plot_inputs_panel(fig, up, um, sp, sm):
    kwargs = {'vmin': 0, 'vmax': 1, 'cmap': CMAP['image']}
    axes = fig.subplots(2, 2, gridspec_kw={'wspace': 0, 'hspace': 0})
    images = (up, um, sp, sm)
    for ax, im in zip(axes.flat, images):
        ax.imshow(im, **kwargs)
    axes[0, 0].set_title('Plastic')
    axes[0, 1].set_title('Metal')
    axes[0, 0].set_ylabel('Uniform')
    axes[1, 0].set_ylabel('Structured')
    # for ax in (axes[0, 1], axes[1, 0]):
    #     plt.text(0.77, 0.05, '2x', transform=ax.transAxes, color='white', fontsize=LARGE_SIZE) # , bbox=dict(facecolor='black', alpha=0.5))
    axes[0, 0].annotate("readout",
        xy=(0.78, 0.30), xycoords='axes fraction',
        xytext=(0.78, 0.04), textcoords='axes fraction',
        horizontalalignment="center", size=SMALL_SIZE,
        arrowprops=dict(facecolor='black', width=0.5, headwidth=4, headlength=3)
        )
    remove_ticks(axes)

def plot_output_panel(fig, input1, input2, map, mask, map_plotter, title):
    kwargs = {'vmin': 0, 'vmax': 1, 'cmap': CMAP['image']}
    gs = gridspec.GridSpec(2, 4, figure=fig, wspace=0.5)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[:, 1:3])
    ax1.imshow(input1, **kwargs)
    ax2.imshow(input2, **kwargs)
    ax3.set_title(title)
    map_plotter(ax3, map, mask)
    ax3.annotate("",
         xy=(-0.04, 0.5), xycoords='axes fraction',
         xytext=(-0.17, 0.5), textcoords='axes fraction',
         arrowprops=dict(facecolor='black', width=0.5, headwidth=4, headlength=3)
         )
    for ax in (ax1, ax2, ax3):
        remove_ticks(ax)
    return ax1, ax2, ax3

root = '/Users/artoews/root/code/projects/metal-phantom/feb2/'
slc = (slice(None), slice(None), 19)

empty_images = np.load(path.join(root, 'artifact', 'images.npy'))
lattice_images = np.load(path.join(root, 'distortion', 'images.npy'))
implant_mask = np.load(path.join(root, 'artifact', 'implant-mask.npy'))
ia_maps = np.load(path.join(root, 'artifact', 'ia-maps.npy'))
gd_maps = np.load(path.join(root, 'distortion', 'gd-maps.npy'))
gd_masks = np.load(path.join(root, 'distortion', 'gd-masks.npy'))
snr_maps = np.load(path.join(root, 'snr', 'snr-maps.npy'))
snr_masks = np.load(path.join(root, 'snr', 'snr-masks.npy'))
res_maps = np.load(path.join(root, 'resolution', 'res-maps.npy'))
res_masks = np.load(path.join(root, 'resolution', 'res-masks.npy'))

up = empty_images[0][slc]
um = empty_images[1][slc]
sp = lattice_images[0][slc]
sm = lattice_images[1][slc]

ia_map = ia_maps[0][slc]
gd_map = -gd_maps[0][..., 0][slc]
gd_mask = gd_masks[1][slc]
snr_map = snr_maps[0][slc]
snr_mask = snr_masks[0][slc]
res_map = res_maps[0][..., 0][slc]
res_mask = (res_map != 0)

fig = plt.figure(figsize=(FIG_WIDTH[2], FIG_WIDTH[2]*0.35))
margin = 0.04
subfigs = fig.subfigures(1, 2, width_ratios=[1, 2], wspace=margin/2, hspace=margin)
subsubfigs = subfigs[1].subfigures(2, 2, wspace=margin/2, hspace=margin)

plot_inputs_panel(subfigs[0], up, um, sp, sm)
plot_output_panel(subsubfigs[0, 0], up, um, ia_map, None, plot_ia_map, 'Intensity Artifact')
plot_output_panel(subsubfigs[0, 1], um, um, snr_map, snr_mask, plot_snr_map, 'SNR')
plot_output_panel(subsubfigs[1, 0], sp, sm, gd_map, gd_mask, plot_gd_map, 'Geometric Distortion')
ax, _, _ = plot_output_panel(subsubfigs[1, 1], sp, sm, res_map, res_mask, plot_res_map, 'Spatial Resolution')

for spine in ax.spines.values():
    spine.set_edgecolor('blue')

color_panels([subfigs[0],] + list(subsubfigs.ravel()))
label_panels([subfigs[0],] + list(subsubfigs.ravel()))

plt.savefig(path.join(root, 'figure2.png'), dpi=DPI)

plt.show()
