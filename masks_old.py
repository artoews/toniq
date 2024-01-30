import numpy as np
import scipy.ndimage as ndi
from skimage import filters, morphology, util

from util import safe_divide


# no longer needed - but make an artifact map function that just takes the difference of plastic and metal, then safe_divides by median filtered plastic image, masking for implant
def get_typical_level(image, mask_signal=None, mask_implant=None, filter_size=10):
    if mask_implant is None:
        mask_empty = get_mask_empty(image)
        mask_implant = get_mask_implant(mask_empty)
    if mask_signal is None:
        mask_signal = get_mask_signal(image)
    footprint = morphology.cube(filter_size)
    signal_sum = ndi.generic_filter(image * mask_signal, np.sum, footprint=footprint)
    signal_count = ndi.generic_filter(mask_signal, np.sum, footprint=footprint, output=float)
    signal_mean =  np.divide(signal_sum, signal_count, out=np.zeros_like(signal_sum), where=signal_count > (filter_size ** 3) / 4)
    signal_mean[signal_mean==0] = np.median(signal_mean[mask_signal]) # fill in remaining implant void
    # can do something more accurate than median?
    # could run the selective mean filter again but only use the result to update the implant region until it’s filled
    # or just run it with a larger filter to begin with as done for SNR
    # the problem is that it can make signal_ref biased by the bright side of the phantom.
    return signal_mean


# unused, leave it for the future to see if it works
def get_mask_lattice(image, diff_size=5, morph_size=10):
    filtered_diffs = []
    for axis in range(image.ndim):
        diff = np.abs(np.diff(image, axis=axis, append=0))
        filtered_diff = ndi.generic_filter(diff, np.sum, footprint=morphology.cube(diff_size))
        filtered_diffs.append(filtered_diff)
    min_diff = np.min(np.stack(filtered_diffs, axis=-1), axis=-1)
    threshold = filters.threshold_otsu(min_diff)  # global Otsu
    mask = min_diff > threshold
    mask = morphology.binary_opening(mask, morphology.cube(morph_size))
    mask = morphology.binary_closing(mask, footprint=morphology.cube(morph_size))
    return mask

# only used as input for implant mask, or where it shouldn't be needed, as for distortion masks (no structure out there!)
def get_mask_empty(image, filter_radius=3):
    image = ndi.median_filter(image, footprint=morphology.ball(filter_radius))  # remove noise & structure
    mask = image < filters.threshold_otsu(image) # global Otsu
    mask = morphology.binary_opening(mask, morphology.ball(filter_radius))  # erosion (min), then dilation (max)
    return mask

# no longer needed 
def get_mask_signal(image, filter_size=5, mask_implant=None):
    if mask_implant is None:
        mask_empty = get_mask_empty(image)
        mask_implant = get_mask_implant(mask_empty)
    image = np.abs(image)
    image = util.img_as_ubyte(image / np.max(image))
    threshold = filters.rank.otsu(image, morphology.cube(filter_size))  # local Otsu
    mask = image > threshold
    # mask = np.ones_like(mask)
    # mask = morphology.binary_erosion(mask, footprint=morphology.ball(1))
    # mask = np.logical_and(mask, ~mask_implant)
    return mask

# can do this much more simply with lattice-free images. just a global otsu threshold and binary_opening will do
def get_mask_implant(mask_empty, verbose=False):
    labels, max_label = morphology.label(mask_empty, return_num=True)
    if verbose:
        print_labels(labels, max_label)
    counts = [np.sum(labels == i) for i in range(max_label+1)]
    order = np.argsort(counts)
    implant = order[-3] # implant is usually the 3rd largest group (after air/frame, oil)
    mask = (labels == implant)
    mask = morphology.binary_dilation(mask, footprint=morphology.ball(1))
    return mask

# unused, and can just use morphology.remove_small_objects for this?
def remove_smaller_than(mask, size):
    labelled_mask, num_labels = morphology.label(mask, return_num=True)
    refined_mask = mask.copy()
    for label in range(num_labels):
        label_count = np.sum(refined_mask[labelled_mask == label])
        if label_count < size:
            refined_mask[labelled_mask == label] = 0
        else:
            print(label_count)
    return refined_mask

# modify this to be a general "combine masks and smooth the edges" function so it can take a variable number of masks and just clean up the union
def get_mask_register(mask_empty, mask_implant, mask_artifact, filter_radius=5):
    mask = (mask_implant + mask_empty + mask_artifact) == 0
    mask = ndi.binary_opening(mask, structure=morphology.ball(filter_radius)) # erosion then dilation
    mask = ndi.binary_closing(mask, structure=morphology.ball(filter_radius)) # dilation then erosion
    return mask

# shouldn't need a function for this - just threshold the artifact map
def get_mask_artifact(reference, target, mask_implant=None, signal_ref=None, thresh=0.3):
    if signal_ref is None:
        if mask_implant is None:
            mask_empty = get_mask_empty(reference)
            mask_implant = get_mask_implant(mask_empty)
        mask_signal = get_mask_signal(reference)
        signal_ref = get_typical_level(reference, mask_signal, mask_implant)
    error = target - reference 
    normalized_error = safe_divide(error, signal_ref)
    mask_artifact, _ = get_mask_extrema(normalized_error, thresh, 'mean', abs_margin=True)
    return mask_artifact

# scrap
def get_mask_artifact_old(error):
    mask, _ = get_mask_extrema(error, 0.3, 'mean', abs_margin=True)
    return mask

# scrap
def get_mask_hyper(error):
    mask, _ = get_mask_extrema(error, 0.3, 'mean', abs_margin=False)
    return mask

# scrap
def get_mask_hypo(error):
    mask, _ = get_mask_extrema(error, -0.3, 'mean', abs_margin=False)
    return mask

# scrap
def get_mask_extrema(error, margin, mode, filter_size=5, abs_margin=True):
    footprint = morphology.cube(filter_size)
    if mode == 'max':
        filtered_error = ndi.maximum_filter(error * np.sign(margin), footprint=footprint)
    elif mode == 'min':
        filtered_error = ndi.minimum_filter(error * np.sign(margin), footprint=footprint)
    elif mode == 'mean':
        filtered_error = ndi.generic_filter(error, np.mean, footprint=footprint)
    elif mode == 'median':
        filtered_error = ndi.median_filter(error, footprint=footprint)
    else:
        raise ValueError('unrecognized mode: {}'.format(mode))
    if abs_margin:
        filtered_error = np.abs(filtered_error)
        mask = filtered_error > np.abs(margin)
    else:
        mask = filtered_error * np.sign(margin) > np.abs(margin)
    return mask, filtered_error

# scrap
def get_all_masks(image_clean, image_distorted, combine=False, denoise=False):

    empty = get_mask_empty(image_clean)
    implant = get_mask_implant(empty)
    signal = get_mask_signal(image_clean)

    error = image_distorted - image_clean 
    if denoise:
        error = denoise(error)

    signal_ref = get_typical_level(image_clean, signal, implant)
    hyper = get_mask_hyper(error, signal_ref)
    hypo = get_mask_hypo(error, signal_ref)
    artifact = get_mask_artifact_old(error, signal_ref)

    out = (implant, empty, hyper, hypo, artifact)

    if combine:
        return combine_masks(*out)
    else:
        return out

# scrap 
def print_labels(labels, max_label):
    for i in range(max_label + 1):
        print('label == {} has size {}'.format(i, np.sum(labels==i)))

# scrap
def combine_masks_2(implant, empty, hyper, hypo, artifact):
    mask = 0 * np.ones(empty.shape)
    mask[artifact] = 3
    mask[hypo] = 2
    mask[hyper] = 4
    mask[empty] = 0
    mask[implant] = 1
    return mask / 4

# scrap
def combine_masks(implant, empty, hyper, hypo, artifact):
    mask = 2 * np.ones(empty.shape)
    mask[artifact] = 3
    mask[hypo] = 4
    mask[hyper] = 5
    mask[empty] = 0
    mask[implant] = 1
    return mask / 5
