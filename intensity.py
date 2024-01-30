import numpy as np
import scipy.ndimage as ndi

from skimage import morphology

def map_snr(image1, image2, mask, filter_size=10, min_coverage=0.25):
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

def noise_std(image1, image2, filter_size=10):
    footprint = morphology.cube(filter_size)
    image_diff = np.abs(image2) - np.abs(image1)
    noise = ndi.generic_filter(image_diff, np.std, footprint=footprint) / np.sqrt(2)
    return noise

def predict_fse_intensity_error(field, profile, pixel_bw=None):
    image = profile(field) # assume uniform image with unit-strength signal
    if pixel_bw is not None:
        pass
        # TODO I have asked how to apply deformation field to transform image here: https://github.com/SuperElastix/elastix/discussions/1016
    error = image - 1
    return error