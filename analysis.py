import numpy as np
import scipy.ndimage as ndi
from skimage import filters, morphology, util

def normalize(image, pct=99):
    return image / np.percentile(image, pct)

def equalize(image, reference, pct=90):
    # TODO find a more principled way
    return image / np.percentile(image, pct) * np.percentile(reference, pct)

def denoise(image, filter_radius=2):
    footprint = morphology.ball(filter_radius)
    return ndi.median_filter(image, footprint=footprint)

def cleanup(mask, filter_radius=2):
    footprint = morphology.ball(filter_radius)
    return morphology.binary_opening(mask, footprint)  # erosion (min), then dilation (max)

def get_mask_empty(image):
    image = denoise(image)
    mask = image < filters.threshold_otsu(image) # global Otsu
    return cleanup(mask)

def get_mask_signal(image1, image2, filter_radius=5):
    image_product = np.abs(image1) * np.abs(image2)
    footprint = morphology.ball(filter_radius)
    image_product = util.img_as_ubyte(image_product / np.max(image_product))
    mask = image_product > filters.rank.otsu(image_product, footprint)  # local Otsu
    return mask

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

def get_mask_extrema(error, signal_ref, margin, is_denoised, mag=False, filter_radius=2, return_stages=False):
    if not is_denoised:
        error = denoise(error)
    if mag:
        error = np.abs(error)
    footprint = morphology.ball(filter_radius)
    max_error = ndi.maximum_filter(error * np.sign(margin), footprint=footprint)
    mask = max_error > np.abs(margin) * signal_ref
    mask_clean = cleanup(mask)
    if return_stages:
        return mask_clean, mask, max_error * np.sign(margin)
    else:
        return mask_clean

def print_labels(labels, max_label):
    for i in range(max_label + 1):
        print('label == {} has size {}'.format(i, np.sum(labels==i)))

def combine_masks(implant, empty, hyper, hypo, artifact):
    mask = np.zeros(empty.shape)
    mask[artifact] = 2
    mask[hypo] = 3
    mask[hyper] = 4
    mask[empty] = 0
    mask[implant] = 1
    return mask / 4

def signal_to_noise(image1, image2, mask_signal, mask_empty, filter_radius=10):
    # "Difference Method" from Reeder et al 2005, extended to include a mask reducing signal bias from lattice
    footprint = morphology.ball(filter_radius)
    image_sum = np.abs(image2) + np.abs(image1)
    image_diff = np.abs(image2) - np.abs(image1)
    filter_sum = ndi.generic_filter(image_sum * mask_signal, np.sum, footprint=footprint)
    filter_count = ndi.generic_filter(mask_signal, np.sum, footprint=footprint, output=float)
    signal =  np.divide(filter_sum, filter_count, out=np.zeros_like(filter_sum), where=filter_count > 0) / 2
    # signal = ndi.generic_filter(image_sum, np.mean, footprint=footprint) / 2
    noise = ndi.generic_filter(image_diff, np.std, footprint=footprint) / np.sqrt(2)
    snr = np.divide(signal * np.logical_not(mask_empty), noise, out=np.zeros_like(signal), where=noise > 0)
    return snr, signal, noise, mask_signal
