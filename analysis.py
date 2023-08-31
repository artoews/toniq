import numpy as np
import scipy.ndimage as ndi
from skimage import filters, morphology
from time import time

def denoise(image, filter_radius=1):
    footprint = morphology.ball(filter_radius)
    return ndi.median_filter(image, footprint=footprint)

def cleanup(mask, filter_radius=2):
    footprint = morphology.ball(filter_radius)
    return morphology.binary_opening(mask, footprint)  # erosion (min), then dilation (max)

def get_mask_empty(image):
    image = denoise(image)
    mask = image < filters.threshold_otsu(image)
    return cleanup(mask)

def get_mask_implant(mask_empty, verbose=False):
    labels, max_label = morphology.label(mask_empty, return_num=True)
    if verbose:
        print_labels(labels, max_label)
    counts = [np.sum(labels == i) for i in range(max_label+1)]
    order = np.argsort(counts)
    implant = order[-3] # implant is 3rd largest group (after air/frame, oil)
    return labels == implant

def get_typical_level(image, filter_radius=3):
    image = denoise(image)
    footprint = morphology.ball(filter_radius)
    image = morphology.closing(image, footprint=footprint) # dilation (max), then erosion (min)
    return image

def get_mask_hypo(error, signal_ref, is_denoised=True):
    return get_mask_extrema(error, signal_ref, -0.6, is_denoised)

def get_mask_hyper(error, signal_ref, is_denoised=True):
    return get_mask_extrema(error, signal_ref, 0.6, is_denoised)

def get_mask_artifact(error, signal_ref, is_denoised=True):
    return get_mask_extrema(error, signal_ref, 0.3, is_denoised, mag=True)

def get_mask_extrema(error, signal_ref, margin, is_denoised, mag=False, filter_radius=2):
    if not is_denoised:
        error = denoise(error)
    if mag:
        error = np.abs(error)
    footprint = morphology.ball(filter_radius)
    max_error = ndi.maximum_filter(error * np.sign(margin), footprint=footprint)
    mask = max_error > np.abs(margin) * signal_ref
    return cleanup(mask)

def print_labels(labels, max_label):
    for i in range(max_label + 1):
        print('label == {} has size {}'.format(i, np.sum(labels==i)))

# def mean_filter(image, size, mode):
#     filter = np.ones((size,) * image.ndim) / (size ** image.ndim)
#     return ndi.correlate(image, filter, mode=mode)
# 
# def signed_max(arr, size, mode):
#     minimum = ndi.minimum_filter(arr, size=size, mode=mode)
#     maximum = ndi.maximum_filter(arr, size=size, mode=mode)
#     sign = (np.abs(maximum) > np.abs(minimum)) * 2 - 1
#     min_max_stack = np.stack((maximum, minimum), axis=-1)
#     return sign * np.max(np.abs(min_max_stack), axis=-1)
