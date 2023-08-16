import numpy as np
import scipy.ndimage as ndi

def mask_signal_void(artifact_image, reference_image=None, threshold=0.7, filter_size=10):
    if reference_image is None:
        reference_image = np.ones_like(artifact_image) * np.mean(artifact_image)
    mask = artifact_image < reference_image * threshold
    mask = ndi.median_filter(mask, size=filter_size)
    return mask


def energy(image, reference, kernel=None, size=None, footprint=None, mode='reflect'):
    energy = (image - reference) ** 2
    if footprint is None:
        if size is None:
            kernel = None
        else:
            footprint = np.ones((size,) * image.ndim)
    if kernel == 'gaussian':
        energy = ndi.gaussian_filter(energy, sigma=size, mode=mode)
    elif kernel == 'median':
        energy = ndi.median_filter(energy, footprint=footprint, mode=mode)
    elif kernel == 'mean':
        energy = ndi.correlate(energy, footprint, mode=mode)
    return energy

# TODO SNR: 1) noise over an ROI, 2) noise over repetitions in a voxel 
# TODO intensity statistics for threshold-based identification of signal loss?

