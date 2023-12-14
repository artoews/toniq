import numpy as np
import scipy.ndimage as ndi

from masks import get_mask_signal
from skimage import morphology

def map_snr(image1, image2, filter_radius=10, mask=None):
    # "Difference Method" from Reeder et al 2005, extended to include a mask reducing signal bias from lattice
    footprint = morphology.ball(filter_radius)
    # footprint = morphology.cube(filter_size)
    image_sum = np.abs(image2) + np.abs(image1)
    image_diff = np.abs(image2) - np.abs(image1)
    if mask is None:
        mask = get_mask_signal(image1)
    filter_sum = ndi.generic_filter(image_sum * mask, np.sum, footprint=footprint)
    filter_count = ndi.generic_filter(mask, np.sum, footprint=footprint, output=float)
    signal =  np.divide(filter_sum, filter_count, out=np.zeros_like(filter_sum), where=filter_count > 0) / 2
    # signal = ndi.generic_filter(image_sum, np.mean, footprint=footprint) / 2
    noise = ndi.generic_filter(image_diff, np.std, footprint=footprint) / np.sqrt(2)
    # snr = np.divide(signal * np.logical_not(mask_empty), noise, out=np.zeros_like(signal), where=noise > 0)
    snr = np.divide(signal, noise, out=np.zeros_like(signal), where=noise > 0)
    return snr, signal, noise

def noise_std(image1, image2, filter_radius=10):
    footprint = morphology.ball(filter_radius)
    image_diff = np.abs(image2) - np.abs(image1)
    noise = ndi.generic_filter(image_diff, np.std, footprint=footprint) / np.sqrt(2)
    return noise