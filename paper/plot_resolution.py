import matplotlib.pyplot as plt

from plot import overlay_mask, colorbar_axis
from plot_params import *

def plot_res_map(ax, res_map, mask, vmin=1, vmax=4, show_cbar=True):
    im = ax.imshow(res_map, cmap=CMAP['resolution'], vmin=vmin, vmax=vmax)
    if mask is not None:
        overlay_mask(ax, ~mask)
    if show_cbar:
        cbar = colorbar(ax, im, 'FWHM (mm)', ticks=[vmin, vmin + (vmax-vmin)/2, vmax])
        # cbar = plt.colorbar(im, cax=colorbar_axis(ax), ticks=[vmin, vmin + (vmax-vmin)/2, vmax])
        return cbar

def colorbar(ax, im, label, offset=0, ticks=[1, 2, 3]):
    cbar = plt.colorbar(im, cax=colorbar_axis(ax, offset=offset), ticks=ticks)
    cbar.set_label(label, size=SMALL_SIZE)
    cbar.ax.tick_params(labelsize=SMALLER_SIZE)
    return cbar