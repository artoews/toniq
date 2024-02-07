import matplotlib.pyplot as plt
from os import path

from plot import imshow2

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

def colorbar(fig, axes, im, lim=0.6):
    cbar = fig.colorbar(im, ax=axes, ticks=[-lim, -lim/2, 0, lim/2, lim], label='Relative Error (%)', extend='both', shrink=0.9)
    cbar.ax.set_yticklabels(['-{:.0f}'.format(lim*100), '-{:.0f}'.format(lim*50), '0', '{:.0f}'.format(lim*50), '{:.0f}'.format(lim*100)])
    return cbar
