import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from os import path

import psf

def box_plots(fwhms, shapes, save_dir=None, figsize=(10, 8), fontsize=18):

    num_trials = len(shapes) - 1

    matrix_shapes = ['{}x{}'.format(shape[0], shape[1]) for shape in shapes[1:]]
    voxel_size_x = [shapes[0][0] / shape[0] for shape in shapes[1:]]
    voxel_size_y = [shapes[0][1] / shape[1] for shape in shapes[1:]]

    expected_fwhm = np.array([psf.sinc_fwhm(shapes[0], shape_i) for shape_i in shapes[1:]])
    expected_fwhm = np.round(expected_fwhm, 2)
    y_ticks = list(set(expected_fwhm.ravel()))
    y_lim = [0, int(np.max(expected_fwhm))+1]

    fwhm_x_nonzero = [fwhms[i][..., 0][fwhms[i][..., 0] > 0] for i in range(num_trials)]
    fwhm_y_nonzero = [fwhms[i][..., 1][fwhms[i][..., 1] > 0] for i in range(num_trials)]

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=figsize)
    ax = axes[0]
    sns.boxplot(fwhm_x_nonzero, ax=ax)
    ax.set_ylim(y_lim)
    ax.set_yticks(y_ticks)
    ax.set_xticks(range(len(matrix_shapes)))
    ax.set_xticklabels('{:.1f}'.format(v) for v in voxel_size_x)
    ax.set_xlabel('Relative Voxel Size in X (voxels)', fontsize=fontsize)
    ax.set_ylabel('Measured FWHM (voxels)', fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    ax.grid(axis='y')

    ax = axes[1]
    sns.boxplot(fwhm_y_nonzero, ax=ax)
    ax.set_ylim(y_lim)
    ax.set_yticks(y_ticks)
    ax.set_xticks(range(len(matrix_shapes)))
    ax.set_xticklabels('{:.1f}'.format(v) for v in voxel_size_y)
    ax.set_xlabel('Relative Voxel Size in Y (voxels)', fontsize=fontsize)
    ax.set_ylabel('Measured FWHM (voxels)', fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    ax.grid(axis='y')

    newax = ax.twiny()
    newax.set_frame_on(True)
    newax.patch.set_visible(False)
    newax.xaxis.set_ticks_position('bottom')
    newax.xaxis.set_label_position('bottom')
    newax.spines['bottom'].set_position(('outward', 70))
    newax.set_xticks(range(len(matrix_shapes)))
    newax.set_xticklabels(matrix_shapes, rotation=30, ha='right', fontsize=fontsize*0.75)
    newax.set_xlabel('Matrix Shape   ', fontsize=fontsize)

    plt.subplots_adjust(hspace=0.3, top=0.95, bottom=0.2)

    if save_dir is not None:
        plt.savefig(path.join(save_dir, 'resolution.png'), dpi=300)