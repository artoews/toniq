import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.transforms as mtransforms
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from os import path, makedirs

from plot_params import *
from plot import remove_ticks

def inputs_panel(fig, up, um, sp, sm):
    kwargs = {'vmin': 0, 'vmax': 1, 'cmap': CMAP['image']}
    axes = fig.subplots(2, 2, gridspec_kw={'wspace': 0, 'hspace': 0})
    images = (up, um, sp, sm)
    for ax, im in zip(axes.flat, images):
        ax.imshow(im, **kwargs)
    axes[0, 0].set_title('Plastic')
    axes[0, 1].set_title('Metal')
    axes[0, 0].set_ylabel('Uniform')
    axes[1, 0].set_ylabel('Structured')
    remove_ticks(axes)

def output_panel(fig, input1, input2, map, map_plotter, title):
    kwargs = {'vmin': 0, 'vmax': 1, 'cmap': CMAP['image']}
    gs = gridspec.GridSpec(2, 4, figure=fig, wspace=0.5)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[:, 1:3])
    ax1.imshow(input1, **kwargs)
    ax2.imshow(input2, **kwargs)
    ax3.set_title(title)
    map_plotter(ax3, map)
    ax3.annotate("",
         xy=(-0.04, 0.5), xycoords='axes fraction',
         xytext=(-0.17, 0.5), textcoords='axes fraction',
         arrowprops=dict(facecolor='black', width=0.5, headwidth=4, headlength=3)
         )
    for ax in (ax1, ax2, ax3):
        remove_ticks(ax)

def colorbar_axis(ax):
    return inset_axes(
        ax,
        width="5%",
        height="100%",
        loc="lower left",
        bbox_to_anchor=(1.05, 0., 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )

def plot_ia(ax, map, lim=0.6):
    im = ax.imshow(map, cmap=CMAP['artifact'], vmin=-lim, vmax=lim)
    cbar = fig.colorbar(im, cax=colorbar_axis(ax), ticks=[-lim, -lim/2, 0, lim/2, lim], label='Relative Error (%)', extend='both')
    cbar.ax.set_yticklabels(['-{:.0f}'.format(lim*100), '-{:.0f}'.format(lim*50), '0', '{:.0f}'.format(lim*50), '{:.0f}'.format(lim*100)])

def ia_panel(fig, input1, input2, map, lim=0.6):
    kwargs = {'vmin': 0, 'vmax': 1, 'cmap': CMAP['image']}
    gs = gridspec.GridSpec(2, 4, figure=fig, wspace=0.5)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[:, 1:3])
    ax1.imshow(input1, **kwargs)
    ax2.imshow(input2, **kwargs)
    ax3.set_title('Intensity Artifact')
    im = ax3.imshow(map, cmap=CMAP['artifact'], vmin=-0.6, vmax=0.6)
    ax3.annotate("",
             xy=(-0.04, 0.5), xycoords='axes fraction',
             xytext=(-0.17, 0.5), textcoords='axes fraction',
             arrowprops=dict(facecolor='black', width=0.5, headwidth=4, headlength=3)
             )
    cbar = fig.colorbar(im, cax=colorbar_axis(ax3), ticks=[-lim, -lim/2, 0, lim/2, lim], label='Relative Error (%)', extend='both')
    cbar.ax.set_yticklabels(['-{:.0f}'.format(lim*100), '-{:.0f}'.format(lim*50), '0', '{:.0f}'.format(lim*50), '{:.0f}'.format(lim*100)])
    for ax in (ax1, ax2, ax3):
        remove_ticks(ax)

def label_panel(subfig, label):
    trans = mtransforms.ScaledTranslation(0.02, -0.12, fig.dpi_scale_trans)
    plt.text(0, 1, label, transform=subfig.transSubfigure + trans)

root = '/Users/artoews/root/code/projects/metal-phantom/feb2/'
slc = (slice(None), slice(None), 19)

empty_images = np.load(path.join(root, 'artifact', 'images.npy'))
lattice_images = np.load(path.join(root, 'distortion', 'images.npy'))
implant_mask = np.load(path.join(root, 'artifact', 'implant-mask.npy'))
ia_maps = np.load(path.join(root, 'artifact', 'ia-maps.npy'))

up = empty_images[0][slc]
um = empty_images[1][slc]
sp = lattice_images[0][slc]
sm = lattice_images[1][slc]

fig = plt.figure(figsize=(FIG_WIDTH[2], FIG_WIDTH[2]*0.35))
margin = 0.04
subfigs = fig.subfigures(1, 2, width_ratios=[1, 2], wspace=margin/2, hspace=margin)
subsubfigs = subfigs[1].subfigures(2, 2, wspace=margin/2, hspace=margin)

inputs_panel(subfigs[0], up, um, sp, sm)
output_panel(subsubfigs[0, 0], up, um, ia_maps[0][slc], plot_ia, 'Intensity Artifact')
output_panel(subsubfigs[0, 1], up, um, ia_maps[0][slc], plot_ia, 'SNR')
output_panel(subsubfigs[1, 0], sp, sm, ia_maps[0][slc], plot_ia, 'Geometric Distortion')
output_panel(subsubfigs[1, 1], sp, sm, ia_maps[0][slc], plot_ia, 'Spatial Resolution')

panel_background = '0.9'
subfigs[0].set_facecolor(panel_background)
for panel in [subfigs[0],] + list(subsubfigs.ravel()):
    panel.set_facecolor(panel_background)

label_panel(subfigs[0], '(A)')
for subfig, label in zip(subsubfigs.flat, ('(B)', '(C)', '(D)', '(E)')):
    label_panel(subfig, label)

plt.savefig(path.join(root, 'figure2.png'), dpi=300)

plt.show()
