import numpy as np
import matplotlib.pyplot as plt


from os import path

from resolution import sinc_fwhm

from plot import overlay_mask, colorbar_axis
from plot_params import *

def box_plots(fwhms, shapes, save_dir=None, figsize=(5, 5)):

    num_trials = len(shapes) - 1

    matrix_shapes = ['{}x{}'.format(shape[0], shape[1]) for shape in shapes[1:]]
    voxel_size_x = [shapes[0][0] / shape[0] for shape in shapes[1:]]
    voxel_size_y = [shapes[0][1] / shape[1] for shape in shapes[1:]]

    expected_fwhm = np.array([sinc_fwhm(shapes[0], shape_i) for shape_i in shapes[1:]])
    expected_fwhm = np.round(expected_fwhm, 1)
    y_ticks = list(set(expected_fwhm.ravel()))
    y_lim = [0, int(np.round(np.max(expected_fwhm)+1))]
    y_ticks += y_lim
    # y_ticks += [1.3, 1.7]

    fwhm_x_nonzero = [fwhms[i][..., 0][fwhms[i][..., 0] > 0] for i in range(num_trials)]
    fwhm_y_nonzero = [fwhms[i][..., 1][fwhms[i][..., 1] > 0] for i in range(num_trials)]

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=figsize)
    ax = axes[0]
    ax.violinplot(fwhm_x_nonzero, showmedians=True)
    ax.set_ylim(y_lim)
    ax.set_yticks(y_ticks)
    ax.set_xticks(np.arange(len(matrix_shapes))+1)
    ax.set_xticklabels('{:.1f}'.format(v) for v in voxel_size_x)
    ax.set_xlabel('Nominal Resolution (relative)')
    ax.set_ylabel('FWHM (relative)')
    # ax.tick_params(labelsize=fontsize)
    ax.grid(axis='y')

    ax = axes[1]
    ax.violinplot(fwhm_y_nonzero, showmedians=True)
    ax.set_ylim(y_lim)
    ax.set_yticks(y_ticks)
    ax.set_xticks(np.arange(len(matrix_shapes))+1)
    ax.set_xticklabels('{:.1f}'.format(v) for v in voxel_size_y)
    ax.set_xlabel('Nominal Resolution (relative)')
    ax.set_ylabel('FWHM (relative)')
    # ax.tick_params(labelsize=fontsize)
    ax.grid(axis='y')

    # newax = ax.twiny()
    # newax.set_frame_on(True)
    # newax.patch.set_visible(False)
    # newax.xaxis.set_ticks_position('bottom')
    # newax.xaxis.set_label_position('bottom')
    # newax.spines['bottom'].set_position(('outward', 70))
    # newax.set_xticks(range(len(matrix_shapes)))
    # newax.set_xticklabels(matrix_shapes, rotation=30, ha='right')
    # newax.set_xlabel('Matrix Shape   ')

    # plt.subplots_adjust(hspace=0.3, left=0.2, top=0.95, bottom=0.3)
    plt.subplots_adjust(hspace=0.3, left=0.2, top=0.95)

    if save_dir is not None:
        plt.savefig(path.join(save_dir, 'fwhm-stats.png'), dpi=300)


def plot_fwhm(fwhms, slc, figsize=(10, 5), vmin=0, vmax=3, save_dir=None):
    fig, axes = plt.subplots(figsize=figsize, nrows=2, ncols=len(fwhms)+1, layout='constrained', width_ratios=[1] * len(fwhms) + [0.1])
    for ax in axes[:, :-1].ravel():
        ax.axis('off')
    for i in range(2):
        for j in range(len(fwhms)):
            im = axes[i, j].imshow(fwhms[j][slc + (i,)], vmin=vmin, vmax=vmax, cmap=CMAP['resolution'])
        fig.colorbar(im, cax=axes[i, -1], ticks=[vmin, 1, 1.7, 2.4, 3.6, vmax], label='FWHM (relative)')
    if save_dir is not None:
        plt.savefig(path.join(save_dir, 'fwhm-images.png'), dpi=300)

def plot_res_map(ax, snr_map, mask, vmin=1, vmax=3, show_cbar=True):
    im = ax.imshow(snr_map, cmap=CMAP['resolution'], vmin=vmin, vmax=vmax)
    if mask is not None:
        overlay_mask(ax, ~mask)
    if show_cbar:
        cbar = plt.colorbar(im, cax=colorbar_axis(ax), ticks=[vmin, vmin + (vmax-vmin)/2, vmax])
        cbar.set_label('FWHM\n(mm, readout)', size=SMALL_SIZE)
        cbar.ax.tick_params(labelsize=SMALLER_SIZE)