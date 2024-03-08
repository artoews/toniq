import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns

from os import path

from plot import imshow2, colorbar_axis, overlay_mask
from plot_params import *

def setup_axes(ax, rbw1, rbw2, fontsize):
    loosely_dashed = (0, (5, 10))
    ax.axline((0, 0), (1, np.sqrt(rbw1 / rbw2)), color='k', linestyle=loosely_dashed)
    # ax.set_xlim([40, 80])
    # ax.set_ylim([20, 60])
    # ax.set_xticks(range(10, 51, 10))
    # ax.set_yticks(range(10, 51, 10))
    ax.set_xlabel('SNR at RBW={}kHz'.format(rbw1), fontsize=fontsize)
    ax.set_ylabel('SNR at RBW={}kHz'.format(rbw2), fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    ax.grid()

def scatter(snrs, rbw, save_dir=None, figsize=(8, 5), fontsize=20):
    fig, ax = plt.subplots(figsize=figsize)
    plt.subplots_adjust(bottom=0.2)
    setup_axes(ax, rbw[0], rbw[1], fontsize*0.8)
    ax.scatter(snrs[0], snrs[1], s=0.01, marker='.')
    if save_dir is not None:
        fig.savefig(path.join(save_dir, 'snr_pixel_cloud.png'), dpi=300)
    return fig, ax

def lines(snrs, rbw, save_dir=None, figsize=(8, 5), fontsize=20):
    fig, ax = plt.subplots(figsize=figsize)
    plt.subplots_adjust(bottom=0.2)
    setup_axes(ax, rbw[0], rbw[1], fontsize*0.8)
    sns.lineplot(x=np.round(snrs[0]).ravel(), y=snrs[1].ravel(), ax=ax)
    if save_dir is not None:
        fig.savefig(path.join(save_dir, 'validation_snr.png'), dpi=300)
    return fig, ax

def demo(image1, image2, signal, noise_std, snr):

    slc_xy = (slice(None), slice(None), image1.shape[2]//2)
    slc_xz = (slice(None), image1.shape[1]//2, slice(None))

    image_sum = (image1 + image2) / 2
    image_diff = image1 - image2
    max_diff = np.max(np.abs(image_diff))

    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(12, 4), layout='constrained')
    gs = axes[0, 0].get_gridspec()
    for i in (0, 1):
        for j in (0, 1, 4):
            axes[i, j].remove()
    ax0 = fig.add_subplot(gs[:, 0])
    ax1 = fig.add_subplot(gs[:, 1])
    ax4 = fig.add_subplot(gs[:, 4])

    imshow2(ax0, image1, slc_xy, slc_xz)
    imshow2(ax1, image2, slc_xy, slc_xz)
    imshow2(axes[0, 2], image_sum, slc_xy, slc_xz)
    imshow2(axes[1, 2], image_diff, slc_xy, slc_xz, vmin=-max_diff, vmax=max_diff)
    imshow2(axes[0, 3], signal, slc_xy, slc_xz)
    imshow2(axes[1, 3], noise_std, slc_xy, slc_xz, vmax=np.max(noise_std))
    imshow2(ax4, snr, slc_xy, slc_xz, vmax=np.max(snr))

    ax0.set_title('Image 1')
    ax1.set_title('Image 2')
    axes[0, 2].set_title('Sum')
    axes[1, 2].set_title('Difference')
    axes[0, 3].set_title('Mean')
    axes[1, 3].set_title('St. Dev.')
    ax4.set_title('SNR')

    return fig

def plot_snr_map(ax, snr_map, mask, show_cbar=True, ticks=[0, 100, 200]):
    # lim = np.round(np.max(snr_map)+4.99, -1)
    im = ax.imshow(snr_map, cmap=CMAP['snr'], vmin=ticks[0], vmax=ticks[-1])
    if mask is not None:
        overlay_mask(ax, ~mask)
    if show_cbar:
        cbar = plt.colorbar(im, cax=colorbar_axis(ax), ticks=ticks)
        cbar.set_label('SNR', size=SMALL_SIZE)
        cbar.ax.tick_params(labelsize=SMALLER_SIZE)
        return cbar

if __name__ == '__main__':

    root_dir = '/Users/artoews/root/code/projects/metal-phantom/sandbox/jan15'
    data = np.load(path.join(root_dir, 'snr', 'outputs.npz'))
    for var in data:
        globals()[var] = data[var]

    i = 0
    fig = demo(images[2*i], images[2*i+1], signals[i], noise_stds[i], snrs[i])
    plt.savefig(path.join(root_dir, 'snr', 'snr-demo.png'), dpi=300)
    plt.show()