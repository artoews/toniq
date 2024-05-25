"""Functions for intensity artifact (IA) mapping & plotting.

"""
import matplotlib.pyplot as plt
import numpy as np
from skimage import morphology

from toniq.filter import nanmean_filter
from toniq.plot import colorbar_axis, overlay_mask
from toniq.plot_params import *
from toniq.util import safe_divide

def get_map(plastic_image, metal_image, implant_mask, filter_size=3):
    """Get IA map.

    The IA map is computed as the normalized difference between images with and without IA (i.e. with and without metal).
    The difference is normalized by a signal reference representing the expected signal for a uniform phantom lacking IA.

    Args:
        plastic_image (array): image with no IA (containing plastic replica implant)
        metal_image (array): image with IA (containing metal implant)
        implant_mask (array): image mask identifying implant region
        filter_size (int, optional): size of filter used for computing the signal reference. Defaults to 3.

    Returns:
        ia_map: IA map
    """
    reference = get_signal_reference(plastic_image, implant_mask, filter_size=filter_size)
    error = metal_image - plastic_image
    ia_map = safe_divide(error, reference)
    return ia_map

def get_signal_reference(image, mask, filter_size=3):
    """Return a denoised version of an image with masked in-filling.

    Image denoising is achieved using a mean filter.
    In-filling is achieved by setting each masked pixel to the median value in the non-masked portion of that slice.

    Args:
        image (array): image with no IA (i.e. no metal)
        mask (array): mask identifying areas to apply in-filling
        filter_size (int, optional): _description_. Defaults to 3.

    Returns:
        reference: signal reference image
    """
    reference = nanmean_filter(image, ~mask, morphology.cube(filter_size))
    for i in range(image.shape[2]):
        reference[..., i][mask[..., i]] = np.nanmedian(image[..., i])
    return reference

def colorbar(ax, im, lim=0.8, offset=0):
    """Plot colorbar for IA map.

    Args:
        ax (Axes): where map is plotted
        im (AxesImage): mappable image returned by imshow for map plot
        lim (float, optional): +/- limit for colorbar range. Defaults to 0.8.
        offset (int, optional): positonal offset of colorbar from map. Defaults to 0.

    Returns:
        cbar: colorbar object
    """
    # cbar = plt.colorbar(im, cax=colorbar_axis(ax, offset=offset), ticks=[-lim, -lim/2, 0, lim/2, lim], extend='both')
    # cbar.ax.set_yticklabels(['-{:.0f}'.format(lim*100), '-{:.0f}'.format(lim*50), '0', '{:.0f}'.format(lim*50), '{:.0f}'.format(lim*100)])
    cbar = plt.colorbar(im, cax=colorbar_axis(ax, offset=offset), ticks=[-lim, -lim/2, 0, lim/2, lim], extend='both')
    cbar.ax.set_yticklabels(['-{:.0f}'.format(lim*100), '-{:.0f}'.format(lim*50), '0', '{:.0f}'.format(lim*50), '{:.0f}'.format(lim*100)])
    cbar.set_label('Relative Error (%)', size=SMALL_SIZE)
    cbar.ax.tick_params(labelsize=SMALLER_SIZE)
    return cbar

def plot_map(ax, ia_map, mask, lim=0.8, show_cbar=True):
    """Plot IA map.

    Args:
        ax (Axes): where map will be plotted
        ia_map (array): IA map
        mask (array): mask identifying areas where map is valid
        lim (float, optional): +/- limit for color range. Defaults to 0.8.
        show_cbar (bool, optional): whether to include a colorbar. Defaults to True.

    Returns:
        cbar: colorbar object
    """
    im = ax.imshow(ia_map, cmap=CMAP['artifact'], vmin=-lim, vmax=lim)
    if mask is not None:
        overlay_mask(ax, ~mask)
    if show_cbar:
        cbar =colorbar(ax, im, lim=lim)
        return cbar
