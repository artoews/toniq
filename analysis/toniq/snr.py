"""Functions for signal-to-noise ratio (SNR) mapping & plotting.

""" 
import matplotlib as mpl
import matplotlib.pyplot as plt 
import numpy as np
import numpy.typing as npt
import scipy.ndimage as ndi

from skimage import morphology

from toniq.plot import colorbar_axis, overlay_mask
from toniq.plot_params import *

def get_map(
        image1: npt.NDArray[np.float64],
        image2: npt.NDArray[np.float64],
        mask: npt.NDArray[np.bool],
        filter_size: int = 10,
        min_coverage: float = 0.5
        ) -> npt.NDArray[np.float64]:
    """Compute SNR map.

    Implements the "Difference Method" from the paper referenced below.
    The method is extended to use a mask removing signal bias from areas lacking signal (as in plastic).

    Reeder SB, Wintersperger BJ, Dietrich O, et al.
    Practical approaches to the evaluation of signal-to-noise ratio performance with parallel imaging:
    Application with cardiac imaging and a 32-channel cardiac coil.
    Magnetic Resonance in Medicine. 2005;54(3):748-754. doi:10.1002/mrm.20636

    Args:
        image1 (npt.NDArray[np.float64]): one of two images in the analysis pair
        image2 (npt.NDArray[np.float64]): other one of two images in the analysis pair
        mask (npt.NDArray[np.bool]): mask excluding image regions lacking signal (as in plastic)
        filter_size (int, optional): size of cube-shaped footprint used for local signal statistics. Defaults to 10.
        min_coverage (float, optional): minimum fraction of pixels in footprint for computing SNR, below which zero is returned. Defaults to 0.5.

    Returns:
        npt.NDArray[np.float64]: SNR map
    """
    footprint = morphology.cube(filter_size)
    image_sum = np.abs(image2) + np.abs(image1)
    image_diff = np.abs(image2) - np.abs(image1)
    filter_sum = ndi.generic_filter(image_sum * mask, np.sum, footprint=footprint)
    filter_count = ndi.generic_filter(mask, np.sum, footprint=footprint, output=float)
    signal =  np.divide(filter_sum, filter_count, out=np.zeros_like(filter_sum), where=filter_count > footprint.size * min_coverage) / 2
    noise = ndi.generic_filter(image_diff, np.std, footprint=footprint) / np.sqrt(2)
    snr = np.divide(signal, noise, out=np.zeros_like(signal), where=noise > 0)
    snr[~mask] = 0
    return snr

def plot_map(
        ax: plt.Axes,
        snr_map: npt.NDArray[np.float64],
        mask: npt.NDArray[np.bool],
        show_cbar: bool = True,
        ticks: list[int] = [0, 80, 160],
        tick_labels: list[str] = None
        ) -> mpl.colorbar.Colorbar | None:
    """Plot SNR map.

    Args:
        ax (plt.Axes): target plot
        snr_map (npt.NDArray[np.float64]): SNR map
        mask (npt.NDArray[np.bool]): mask identifying areas where map is valid
        show_cbar (bool, optional): whether to include a colorbar. Defaults to True.
        ticks (list[int], optional): colorbar ticks; min/max values are also used to set the color range of plot. Defaults to [0, 80, 160].
        tick_labels (list[str], optional): custom labels corresponding to colorbar ticks. Defaults to None.

    Returns:
        mpl.colorbar.Colorbar: colorbar
    """
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
    else:
        return None
