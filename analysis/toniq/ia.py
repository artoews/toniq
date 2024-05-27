"""Functions for intensity artifact (IA) mapping & plotting.

"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.ndimage as ndi
import warnings
from skimage import morphology

from toniq.plot import colorbar_axis, overlay_mask
from toniq.plot_params import *
from toniq.util import safe_divide, masked_copy

def get_map(
        plastic_image: npt.NDArray[np.float64],
        metal_image: npt.NDArray[np.float64],
        implant_mask: npt.NDArray[np.bool],
        filter_size: int = 3
        ) -> npt.NDArray[np.float64]:
    """Compute IA map.

    The IA map is computed as the normalized difference between images with and without IA (i.e. with and without metal).
    The difference is normalized by a signal reference representing the expected signal for a uniform phantom lacking IA.

    Args:
        plastic_image (npt.NDArray): image with no IA (no metal)
        metal_image (npt.NDArray): image with IA (from metal)
        implant_mask (npt.NDArray): image mask excluding the implant region
        filter_size (int, optional): size of filter used for computing the signal reference. Defaults to 3.

    Returns:
        npt.NDArray[np.float64]: IA map
    """
    reference = get_signal_reference(plastic_image, implant_mask, filter_size=filter_size)
    error = metal_image - plastic_image
    ia_map = safe_divide(error, reference)
    return ia_map

def get_signal_reference(
        image: npt.NDArray[np.float64],
        mask: npt.NDArray[np.bool],
        filter_size: int = 3
        ) -> npt.NDArray[np.float64]:
    """Return a denoised version of an image with masked in-filling.

    Image denoising is achieved using a mean filter.
    In-filling is achieved by setting each masked pixel to the median value in the non-masked portion of that slice.

    Args:
        image (npt.NDArray[np.float64]): image with no IA (i.e. no metal)
        mask (npt.NDArray[np.bool]): mask excluding areas where in-filling will be used
        filter_size (int, optional): size of cube-shaped footprint used for mean filter. Defaults to 3.

    Returns:
        npt.NDArray[np.float64]: processed image
    """
    with warnings.catch_warnings():
        # suppress warning "RuntimeWarning: Mean of empty slice" when footprint covers entirely nan values
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        masked_image = masked_copy(image, mask, fill_val=np.nan)
        reference = ndi.generic_filter(masked_image, np.nanmean, footprint=morphology.cube(filter_size))
    for i in range(image.shape[2]):
        reference[..., i][~mask[..., i]] = np.nanmedian(image[..., i])
    return reference

def colorbar(
        ax: plt.Axes,
        im: mpl.image.AxesImage,
        lim: float = 0.8,
        offset: float = 0
        ) -> mpl.colorbar.Colorbar:
    """Plot colorbar for IA map.

    Args:
        ax (plt.Axes): where map is plotted
        im (mpl.image.AxesImage): mappable image from IA plot
        lim (float, optional): +/- limit for colorbar range. Defaults to 0.8.
        offset (float, optional): positional offset of colorbar from IA map. Defaults to 0.

    Returns:
        mpl.colorbar.Colorbar: colorbar
    """
    cbar = plt.colorbar(im, cax=colorbar_axis(ax, offset=offset), ticks=[-lim, -lim/2, 0, lim/2, lim], extend='both')
    cbar.ax.set_yticklabels(['-{:.0f}'.format(lim*100), '-{:.0f}'.format(lim*50), '0', '{:.0f}'.format(lim*50), '{:.0f}'.format(lim*100)])
    cbar.set_label('Relative Error (%)', size=SMALL_SIZE)
    cbar.ax.tick_params(labelsize=SMALLER_SIZE)
    return cbar

def plot_map(
        ax: plt.Axes,
        ia_map: npt.NDArray[np.float64],
        mask: npt.NDArray[np.bool],
        lim: float = 0.8,
        show_cbar: bool = True
        ) -> mpl.colorbar.Colorbar | None:
    """Plot IA map.

    Args:
        ax (plt.Axes): target plot 
        ia_map (npt.NDArray[np.float64]): IA map
        mask (npt.NDArray[np.bool]): mask identifying areas where map is valid
        lim (float, optional): +/- limit for color range. Defaults to 0.8.
        show_cbar (np.bool, optional): whether to include a colorbar. Defaults to True.

    Returns:
        mpl.colorbar.Colorbar: colorbar
    """
    im = ax.imshow(ia_map, cmap=CMAP['artifact'], vmin=-lim, vmax=lim)
    if mask is not None:
        overlay_mask(ax, ~mask)
    if show_cbar:
        cbar = colorbar(ax, im, lim=lim)
        return cbar
    else:
        return None
