import matplotlib.pyplot as plt
import numpy as np
from skimage import morphology

from filter import nanmean_filter
from plot import colorbar_axis, overlay_mask
from plot_params import *
from util import safe_divide

def get_map(plastic_image, metal_image, implant_mask, filter_size=3):
    reference = get_signal_reference(plastic_image, implant_mask, filter_size=filter_size)
    error = metal_image - plastic_image
    artifact_map = safe_divide(error, reference)
    return artifact_map

def get_signal_reference(plastic_image, implant_mask, filter_size=3):
    reference = nanmean_filter(plastic_image, ~implant_mask, morphology.cube(filter_size))
    for i in range(plastic_image.shape[2]):
        reference[..., i][implant_mask[..., i]] = np.nanmedian(plastic_image[..., i])
    return reference

def colorbar(ax, im, lim=0.8, offset=0):
    # cbar = plt.colorbar(im, cax=colorbar_axis(ax, offset=offset), ticks=[-lim, -lim/2, 0, lim/2, lim], extend='both')
    # cbar.ax.set_yticklabels(['-{:.0f}'.format(lim*100), '-{:.0f}'.format(lim*50), '0', '{:.0f}'.format(lim*50), '{:.0f}'.format(lim*100)])
    cbar = plt.colorbar(im, cax=colorbar_axis(ax, offset=offset), ticks=[-lim, -lim/2, 0, lim/2, lim], extend='both')
    cbar.ax.set_yticklabels(['-{:.0f}'.format(lim*100), '-{:.0f}'.format(lim*50), '0', '{:.0f}'.format(lim*50), '{:.0f}'.format(lim*100)])
    cbar.set_label('Relative Error (%)', size=SMALL_SIZE)
    cbar.ax.tick_params(labelsize=SMALLER_SIZE)
    return cbar

def plot_map(ax, ia_map, mask, lim=0.8, show_cbar=True):
    im = ax.imshow(ia_map, cmap=CMAP['artifact'], vmin=-lim, vmax=lim)
    if mask is not None:
        overlay_mask(ax, ~mask)
    if show_cbar:
        cbar =colorbar(ax, im, lim=lim)
        return cbar
