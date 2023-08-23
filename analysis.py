import numpy as np
import scipy.ndimage as ndi
from time import time

def mask_signal_void(artifact_image, reference_image=None, threshold=0.7, filter_size=10):
    if reference_image is None:
        reference_image = np.ones_like(artifact_image) * np.mean(artifact_image)
    mask = artifact_image < reference_image * threshold
    t0 = time()
    mask = ndi.median_filter(mask, size=filter_size)
    print(time() - t0)
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
    elif kernel == 'max':
        energy = ndi.maximum_filter(energy, footprint=footprint, mode=mode)
    elif kernel == 'mean':
        energy = ndi.correlate(energy, footprint, mode=mode)
    elif kernel == 'kolind':
        energy = np.sum(energy)
    return energy
