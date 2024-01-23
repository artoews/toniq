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

def plot_artifact_results(images, maps_artifact, signal_refs, rbw, save_dir=None):
    shape = images[0].shape
    num_trials = len(maps_artifact)
    slc_xy = (slice(None), slice(None), shape[2] // 2)
    slc_xz = (slice(None), shape[1] // 2, slice(None))
    fs = 20
    fig = plt.figure(figsize=(18, 6), layout='constrained')
    axes = fig.subplots(nrows=num_trials, ncols=4)

    if num_trials == 1:
        axes = axes[None, :]
    axes[0, 0].set_title('Plastic', fontsize=fs)
    axes[0, 1].set_title('Metal', fontsize=fs)
    axes[0, 2].set_title('Error', fontsize=fs)
    axes[0, 3].set_title('Filtered Error', fontsize=fs)
    for i in range(num_trials):
        axes[i, 0].set_ylabel('RBW={:.3g}kHz'.format(rbw[i]), fontsize=fs)
        error = safe_divide(images[2*i+1] - images[2*i], signal_refs[i])
        imshow2(axes[i, 0], images[2*i], slc_xy, slc_xz)
        imshow2(axes[i, 1], images[2*i+1], slc_xy, slc_xz)
        im, _ = imshow2(axes[i, 2], error, slc_xy, slc_xz, cmap='RdBu_r', vmin=-1, vmax=1)
        imshow2(axes[i, 3], maps_artifact[i], slc_xy, slc_xz, cmap='RdBu_r', vmin=-1, vmax=1)
    cbar = fig.colorbar(im, ax=axes[:, 3], ticks=[-1, -0.5, 0, 0.5, 1], label='Relative Error')
    # cbar = plt.colorbar(im, cax=axes[:, 4], ticks=[-1, -0.5, 0, 0.5, 1])
    cbar.ax.set_yticklabels(['-100%', '-50%', '0', '50%', '+100%'])
    # axes[:, 4].tick_params(labelsize=fs*0.75)
    if save_dir is not None:
        plt.savefig(path.join(save_dir, 'validation_artifact.png'), dpi=300)
