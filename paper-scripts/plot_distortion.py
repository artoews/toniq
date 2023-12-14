import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from os import path

from distortion import net_pixel_bandwidth
from plot import overlay_mask
from util import masked_copy

def image_results(images, masks_register, results_masked, rbw, fontsize=20, save_dir=None):
    ''' abstract validation figure panel A: image result '''
    num_trials = len(rbw) - 1
    fixed_image = images[1]
    fixed_mask = masks_register[1]
    fixed_image_masked = masked_copy(fixed_image, fixed_mask)
    fig, axes = plt.subplots(nrows=num_trials, ncols=5, figsize=(20, 8))
    if num_trials == 1:
        axes = axes[None, :]
    fontsize = 20
    kwargs = {'cmap': 'gray', 'vmin': 0, 'vmax': 1}
    axes[0, 0].imshow(fixed_image_masked, **kwargs)
    overlay_mask(axes[0, 0], ~fixed_mask)
    error_multiplier = 3
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    for i in range(num_trials):
        moving_mask = masks_register[1+i]
        moving_image_masked = masked_copy(images[2+i], moving_mask)
        init_error = np.abs(moving_image_masked - fixed_image_masked) * error_multiplier
        result_error = np.abs(results_masked[i] - fixed_image_masked) * error_multiplier
        init_mask =  (moving_image_masked != 0) * (fixed_image_masked != 0)
        result_mask = (results_masked[i] != 0)
        # color_mask = np.zeros(result_error.shape + (4,), dtype=np.uint8)
        # color_mask[~result_mask, :] = np.array([0, 0, 255, 255], dtype=np.uint8)
        axes[i, 1].imshow(moving_image_masked, **kwargs)
        axes[i, 2].imshow(results_masked[i], **kwargs)
        axes[i, 3].imshow(init_error * init_mask, **kwargs)
        axes[i, 4].imshow(result_error * result_mask, **kwargs)
        overlay_mask(axes[i, 1], ~moving_mask)
        overlay_mask(axes[i, 2], ~result_mask)
        overlay_mask(axes[i, 3], ~init_mask)
        overlay_mask(axes[i, 4], ~result_mask)
        axes[i, 1].set_ylabel('RBW={:.3g}kHz'.format(rbw[1+i]), fontsize=fontsize)
        if i > 0:
            plt.delaxes(axes[i, 0])
    axes[0, 0].set_title('Fixed Image', fontsize=fontsize)
    axes[0, 1].set_title('Moving Image', fontsize=fontsize)
    axes[0, 2].set_title('Registration', fontsize=fontsize)
    axes[0, 3].set_title('Initial Error (3x)', fontsize=fontsize)
    axes[0, 4].set_title('Final Error (3x)', fontsize=fontsize)
    axes[0, 0].set_ylabel('RBW={:.3g}kHz'.format(rbw[0]), fontsize=fontsize)
    if save_dir is not None:
        plt.savefig(path.join(save_dir, 'validation_distortion_images.png'), dpi=300)
    return fig, axes

def field_results(true_field, deformation_fields, results_masked, rbw, pbw, fontsize=20, save_dir=None):
    ''' abstract validation figure panel B: field result '''
    num_trials = len(pbw) - 1
    fig, axes = plt.subplots(nrows=num_trials, ncols=4, figsize=(12, 8), gridspec_kw={'width_ratios': [1, 1, 1, 0.1]})
    kwargs = {'cmap': 'RdBu_r', 'vmin': -4, 'vmax': 4}
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    if num_trials == 1:
        axes = axes[None, :]
    for i in range(num_trials):
        if pbw[1+i] == pbw[0]:
            continue
        net_pbw = net_pixel_bandwidth(pbw[1+i], pbw[0])  # Hz
        result_mask = (results_masked[i] != 0)
        simulated_deformation = true_field * 1000 / net_pbw
        measured_deformation = -deformation_fields[i][..., 0]
        axes[i, 0].imshow(simulated_deformation * result_mask, **kwargs)
        axes[i, 1].imshow(measured_deformation * result_mask, **kwargs)
        im = axes[i, 2].imshow((simulated_deformation - measured_deformation) * result_mask, **kwargs)
        overlay_mask(axes[i, 0], ~result_mask)
        overlay_mask(axes[i, 1], ~result_mask)
        overlay_mask(axes[i, 2], ~result_mask)
        axes[i, 0].set_ylabel('RBW={:.3g}kHz'.format(rbw[1+i]), fontsize=fontsize)
        cb = plt.colorbar(im, cax=axes[i, 3], ticks=[-4, -2, 0, 2, 4])
        axes[i, 3].tick_params(labelsize=fontsize*0.75)
        cb.set_label(label='Displacement (pixels)', size=fontsize)
    axes[0, 0].set_title('Reference', fontsize=fontsize)
    axes[0, 1].set_title('Registration', fontsize=fontsize)
    axes[0, 2].set_title('Error', fontsize=fontsize)
    if save_dir is not None:
        plt.savefig(path.join(save_dir, 'validation_distortion_fields.png'), dpi=300)
    return fig, axes

def summary_results(true_field, deformation_fields, results_masked, rbw, pbw, fontsize=20, save_dir=None):
    ''' abstract validation figure panel C: line plots '''
    num_trials = len(pbw) - 1
    fig, ax = plt.subplots(figsize=(8, 7))
    f_max = 1.5
    # colors = ['black', 'red', 'blue']
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    styles = ['dotted', 'solid', 'dashed']
    loosely_dashed = (0, (5, 10))
    for i in range(num_trials):
        if pbw[1+i] == pbw[0]:
            continue
        result_mask = (results_masked[i] != 0)
        net_pbw = net_pixel_bandwidth(pbw[1+i], pbw[0]) / 1000 # kHz
        measured_deformation = -deformation_fields[i][..., 0]
        field_bins = np.round(true_field * 10) / 10
        # measured_deformation = np.abs(measured_deformation)
        # field_bins = np.abs(field_bins)
        # plots mean line and 95% confidence band
        sns.lineplot(x=(field_bins * result_mask).ravel(),
                     y=(measured_deformation * result_mask).ravel(),
                     ax=ax, legend='brief', label='RBW={0:.3g}kHz'.format(rbw[i+1]), color=colors[i], linestyle=styles[i])
        # ax.scatter((field_bins * result_mask).ravel(), (np.abs(measured_deformation) * result_mask).ravel(), c=colors[i], s=0.1, marker='.')
        ax.axline((-f_max, -f_max / net_pbw), (f_max, f_max / net_pbw), color=colors[i], linestyle=loosely_dashed)
        ax.set_xlim([-f_max, f_max])
        ax.set_ylim([-f_max / net_pbw, f_max / net_pbw])
    ax.set_xlabel('Off-Resonance (kHz)', fontsize=fontsize)
    ax.set_ylabel('Displacement (pixels)', fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.grid()
    if save_dir is None:
        plt.savefig(path.join(save_dir, 'validation_distortion_summary.png'), dpi=300)
    return fig, ax
