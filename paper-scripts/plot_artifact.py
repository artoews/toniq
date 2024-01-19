import numpy as np
import matplotlib.pyplot as plt
from os import path

from plot import plotVolumes, overlay_mask, imshow2
from util import safe_divide

def plot_progression(reference, target, map_artifact, signal_ref):
    error = target - reference 
    normalized_error = safe_divide(error, signal_ref)
    map_artifact = map_artifact
    mask_artifact = np.zeros_like(map_artifact)
    mask_artifact[map_artifact > 0.3] = 0.3
    mask_artifact[map_artifact < -0.3] = -0.3
    volumes = (reference - 0.5,
               target - 0.5,
               error,
               normalized_error,
               map_artifact,
               mask_artifact
               )
    titles = ('plastic', 'metal', 'diff', 'norm. diff.', 'filtered norm. diff.', 'threshold at +/-30%')
    fig, tracker = plotVolumes(volumes, 1, len(volumes), titles=titles, figsize=(16, 8), vmin=-1, vmax=1)
    return fig, tracker

def plot_artifact_results(images, maps_artifact, signal_ref, rbw, save_dir=None):
    image_ref = images[0]
    num_trials = len(maps_artifact)
    slc_xy = (slice(None), slice(None), image_ref.shape[2] // 2)
    slc_xz = (slice(None), image_ref.shape[1] // 2, slice(None))
    fs = 20
    fig, axes = plt.subplots(nrows=num_trials, ncols=5, figsize=(18, 6), gridspec_kw={'width_ratios': [1, 1, 1, 1, 0.1]})

    if num_trials == 1:
        axes = axes[None, :]
    imshow2(axes[0, 0], image_ref, slc_xy, slc_xz)
    # axes[0, 0].imshow(image_ref[slc_xy], cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('Plastic', fontsize=fs)
    axes[0, 1].set_title('Metal', fontsize=fs)
    axes[0, 2].set_title('Relative Error', fontsize=fs)
    axes[0, 3].set_title('Mean Filter', fontsize=fs)
    for i in range(num_trials):
        image_i = images[1+i]
        error_i = image_i - image_ref
        normalized_error_i = safe_divide(error_i, signal_ref)
        mask_artifact = np.zeros_like(maps_artifact[i])
        mask_artifact[maps_artifact[i] > 0.3] = 0.3
        mask_artifact[maps_artifact[i] < -0.3] = -0.3
        if i > 0: plt.delaxes(axes[i, 0])
        # axes[i, 1].imshow(image_i[slc_xy], cmap='gray', vmin=0, vmax=1)
        # im2 = axes[i, 2].imshow(normalized_error_i[slc_xy], cmap='RdBu_r', vmin=-1, vmax=1)
        # axes[i, 3].imshow(maps_artifact[i][slc_xy], cmap='RdBu_r', vmin=-1, vmax=1)
        axes[i, 1].set_ylabel('RBW={:.3g}kHz'.format(rbw[i]), fontsize=fs)
        imshow2(axes[i, 1], image_i, slc_xy, slc_xz)
        im, _ = imshow2(axes[i, 2], normalized_error_i, slc_xy, slc_xz, cmap='RdBu_r', vmin=-1, vmax=1)
        imshow2(axes[i, 3], maps_artifact[i], slc_xy, slc_xz, cmap='RdBu_r', vmin=-1, vmax=1)
        plt.colorbar(im, cax=axes[i, 4], ticks=[-1, 0, 1])
        axes[i, 4].tick_params(labelsize=fs*0.75)
    if save_dir is not None:
        plt.savefig(path.join(save_dir, 'validation_artifact.png'), dpi=300)

def plot_artifact_results_overlay(images, maps_artifact, signal_ref, rbw, save_dir=None):
    image_ref = images[0]
    num_trials = len(maps_artifact)
    slc = (slice(None), slice(None), image_ref.shape[2] // 2)
    fs = 20
    fig, axes = plt.subplots(nrows=num_trials, ncols=6, figsize=(18, 6), gridspec_kw={'width_ratios': [1, 1, 1, 1, 1, 0.1]})
    if num_trials == 1:
        axes = axes[None, :]
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0, 0].imshow(image_ref[slc], cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('Plastic', fontsize=fs)
    axes[0, 1].set_title('Metal', fontsize=fs)
    axes[0, 3].set_title('Relative Error', fontsize=fs)
    axes[0, 4].set_title('+ Mean Filter', fontsize=fs)
    axes[0, 2].set_title('Metal + Filtered Error', fontsize=fs)
    # axes[0, 3].set_title('Intensity Artifact Map')
    for i in range(num_trials):
        image_i = images[1+i]
        error_i = image_i - image_ref
        normalized_error_i = safe_divide(error_i, signal_ref)
        mask_artifact = np.zeros_like(maps_artifact[i])
        mask_artifact[maps_artifact[i] > 0.3] = 0.3
        mask_artifact[maps_artifact[i] < -0.3] = -0.3
        if i > 0: plt.delaxes(axes[i, 0])
        im1 = axes[i, 1].imshow(image_i[slc], cmap='gray', vmin=0, vmax=1)
        im2 = axes[i, 3].imshow(normalized_error_i[slc], cmap='RdBu_r', vmin=-1, vmax=1)
        axes[i, 4].imshow(maps_artifact[i][slc], cmap='RdBu_r', vmin=-1, vmax=1)
        axes[i, 2].imshow(image_i[slc], cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_ylabel('RBW={:.3g}kHz'.format(rbw[i]), fontsize=fs)
        plt.colorbar(im2, cax=axes[i, 5], ticks=[-1, 0, 1])
        axes[i, 5].tick_params(labelsize=fs*0.75)
        overlay_mask(axes[i, 2], np.abs(mask_artifact[slc]) >= 0.3, color=[0, 0, 255], alpha=50)
    if save_dir is not None:
        plt.savefig(path.join(save_dir, 'validation_artifact_overlay.png'), dpi=300) 