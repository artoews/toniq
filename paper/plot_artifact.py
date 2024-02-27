import matplotlib.pyplot as plt
from os import path

from plot import imshow2, colorbar_axis, overlay_mask

from plot_params import *

def plot_artifact_results(images, maps_artifact, save_dir=None, lim=0.6):
    shape = images[0].shape
    num_trials = len(maps_artifact)
    slc_xy = (slice(None), slice(None), shape[2] // 2)
    slc_xz = (slice(None), shape[1] // 2, slice(None))
    fig = plt.figure(figsize=(12, 6), layout='constrained')
    axes = fig.subplots(nrows=num_trials, ncols=3)
    if num_trials == 1:
        axes = axes[None, :]
    axes[0, 0].set_title('Plastic')
    axes[0, 1].set_title('Metal')
    axes[0, 2].set_title('Intensity Artifact')
    for i in range(num_trials):
        imshow2(axes[i, 0], images[2*i], slc_xy, slc_xz, y_label='Read', x1_label='Phase', x2_label='Slice')
        imshow2(axes[i, 1], images[2*i+1], slc_xy, slc_xz)
        im, _ = imshow2(axes[i, 2], maps_artifact[i], slc_xy, slc_xz, cmap=CMAP['artifact'], vmin=-lim, vmax=lim)
        colorbar(fig, axes[i, 2], im)
    if save_dir is not None:
        plt.savefig(path.join(save_dir, 'ia_results.png'), dpi=300)

def plot_signal_ref(images, sig_refs, save_dir=None):
    shape = images[0].shape
    num_trials = len(sig_refs)
    slc_xy = (slice(None), slice(None), shape[2] // 2)
    slc_xz = (slice(None), shape[1] // 2, slice(None))
    fig, axes = plt.subplots(figsize=(10, 5), nrows=num_trials, ncols=3)
    for i in range(num_trials):
        imshow2(axes[i, 0], images[2*i], slc_xy, slc_xz, y_label='Read', x1_label='Phase', x2_label='Slice')
        imshow2(axes[i, 1], images[2*i+1], slc_xy, slc_xz, y_label='Read', x1_label='Phase', x2_label='Slice')
        imshow2(axes[i, 2], sig_refs[i], slc_xy, slc_xz, y_label='Read', x1_label='Phase', x2_label='Slice')
    axes[0, 0].set_title('Plastic')
    axes[0, 1].set_title('Metal')
    axes[0, 2].set_title('Reference')
    if save_dir is not None:
        plt.savefig(path.join(save_dir, 'signal_reference.png'), dpi=300)

def colorbar(fig, axes, im, lim=0.6):
    cbar = fig.colorbar(im, ax=axes, ticks=[-lim, -lim/2, 0, lim/2, lim], label='Relative Error (%)', extend='both', shrink=0.9)
    cbar.ax.set_yticklabels(['-{:.0f}'.format(lim*100), '-{:.0f}'.format(lim*50), '0', '{:.0f}'.format(lim*50), '{:.0f}'.format(lim*100)])
    return cbar

def plot_ia_map(ax, ia_map, mask, lim=0.6, show_cbar=True):
    im = ax.imshow(ia_map, cmap=CMAP['artifact'], vmin=-lim, vmax=lim)
    if mask is not None:
        overlay_mask(ax, ~mask)
    if show_cbar:
        cbar = plt.colorbar(im, cax=colorbar_axis(ax), ticks=[-lim, -lim/2, 0, lim/2, lim], extend='both')
        cbar.ax.set_yticklabels(['-{:.0f}'.format(lim*100), '-{:.0f}'.format(lim*50), '0', '{:.0f}'.format(lim*50), '{:.0f}'.format(lim*100)])
        cbar.set_label('Relative Error\n(%)', size=SMALL_SIZE)
        cbar.ax.tick_params(labelsize=SMALLER_SIZE)
