import matplotlib.pyplot as plt 
import numpy as np
from os import path
import scipy.ndimage as ndi
import seaborn as sns
from skimage import morphology

from plot import colorbar_axis, overlay_mask
from plot_params import *

def get_map(image1, image2, mask, filter_size=10, min_coverage=0.5):
    # "Difference Method" from Reeder et al 2005, extended to include a mask reducing signal bias from lattice
    footprint = morphology.cube(filter_size)
    image_sum = np.abs(image2) + np.abs(image1)
    image_diff = np.abs(image2) - np.abs(image1)
    filter_sum = ndi.generic_filter(image_sum * mask, np.sum, footprint=footprint)
    filter_count = ndi.generic_filter(mask, np.sum, footprint=footprint, output=float)
    signal =  np.divide(filter_sum, filter_count, out=np.zeros_like(filter_sum), where=filter_count > footprint.size * min_coverage) / 2
    noise = ndi.generic_filter(image_diff, np.std, footprint=footprint) / np.sqrt(2)
    snr = np.divide(signal, noise, out=np.zeros_like(signal), where=noise > 0)
    snr[~mask] = 0
    return snr, signal, noise

def plot_map(ax, snr_map, mask, show_cbar=True, ticks=[0, 80, 160], tick_labels=None):
    # lim = np.round(np.max(snr_map)+4.99, -1)
    im = ax.imshow(snr_map, cmap=CMAP['snr'], vmin=ticks[0], vmax=ticks[-1])
    if mask is not None:
        overlay_mask(ax, ~mask)
    if show_cbar:
        cbar = plt.colorbar(im, cax=colorbar_axis(ax), ticks=ticks)
        cbar.set_label('SNR', size=SMALL_SIZE)
        if tick_labels is not None:
            cbar.ax.set_yticklabels(tick_labels)
        cbar.ax.tick_params(labelsize=SMALLER_SIZE)
        return cbar

# def setup_axes(ax, rbw1, rbw2, fontsize):
#     loosely_dashed = (0, (5, 10))
#     ax.axline((0, 0), (1, np.sqrt(rbw1 / rbw2)), color='k', linestyle=loosely_dashed)
#     # ax.set_xlim([40, 80])
#     # ax.set_ylim([20, 60])
#     # ax.set_xticks(range(10, 51, 10))
#     # ax.set_yticks(range(10, 51, 10))
#     ax.set_xlabel('SNR at RBW={}kHz'.format(rbw1), fontsize=fontsize)
#     ax.set_ylabel('SNR at RBW={}kHz'.format(rbw2), fontsize=fontsize)
#     ax.tick_params(labelsize=fontsize)
#     ax.grid()
# 
# def scatter(snrs, rbw, save_dir=None, figsize=(8, 5), fontsize=20):
#     fig, ax = plt.subplots(figsize=figsize)
#     plt.subplots_adjust(bottom=0.2)
#     setup_axes(ax, rbw[0], rbw[1], fontsize*0.8)
#     ax.scatter(snrs[0], snrs[1], s=0.01, marker='.')
#     if save_dir is not None:
#         fig.savefig(path.join(save_dir, 'snr_pixel_cloud.png'), dpi=300)
#     return fig, ax
# 
# def lines(snrs, rbw, save_dir=None, figsize=(8, 5), fontsize=20):
#     fig, ax = plt.subplots(figsize=figsize)
#     plt.subplots_adjust(bottom=0.2)
#     setup_axes(ax, rbw[0], rbw[1], fontsize*0.8)
#     sns.lineplot(x=np.round(snrs[0]).ravel(), y=snrs[1].ravel(), ax=ax)
#     if save_dir is not None:
#         fig.savefig(path.join(save_dir, 'validation_snr.png'), dpi=300)
#     return fig, ax
