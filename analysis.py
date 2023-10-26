import numpy as np
import scipy.ndimage as ndi
from skimage import filters, morphology, restoration, util

import register

def normalize(image, pct=99):
    return image / np.percentile(image, pct)

def equalize(image, reference, pct=90):
    # TODO find a more principled way
    return image / np.percentile(image, pct) * np.percentile(reference, pct)

def get_mask_lattice(image, filter_size=5):
    footprint = morphology.cube(filter_size)
    filtered_diffs = []
    for axis in range(image.ndim):
        diff = np.abs(np.diff(image, axis=axis, append=0))
        filtered_diff = ndi.generic_filter(diff, np.sum, footprint=footprint)
        filtered_diffs.append(filtered_diff)
    min_diff = np.min(np.stack(filtered_diffs, axis=-1), axis=-1)
    threshold = filters.threshold_otsu(min_diff)  # global Otsu
    mask = min_diff > threshold
    mask = morphology.binary_opening(mask, footprint)
    mask = morphology.binary_closing(mask, footprint=footprint)
    return mask

def get_mask_empty(image, filter_radius=3):
    image = ndi.median_filter(image, footprint=morphology.ball(filter_radius))  # remove noise & structure
    mask = image < filters.threshold_otsu(image) # global Otsu
    mask = morphology.binary_opening(mask, morphology.ball(filter_radius))  # erosion (min), then dilation (max)
    return mask

def get_mask_signal(image, filter_size=5):
    image = np.abs(image)
    image = util.img_as_ubyte(image / np.max(image))
    threshold = filters.rank.otsu(image, morphology.cube(filter_size))  # local Otsu
    mask = image > threshold
    return mask

def get_mask_implant(mask_empty, verbose=False):
    labels, max_label = morphology.label(mask_empty, return_num=True)
    if verbose:
        print_labels(labels, max_label)
    counts = [np.sum(labels == i) for i in range(max_label+1)]
    order = np.argsort(counts)
    implant = order[-3] # implant is 3rd largest group (after air/frame, oil)
    return labels == implant

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

def get_typical_level(image, signal_mask, implant_mask, filter_size=5):
    # fill in implant area
    filled_image = np.abs(image)
    median_signal = np.median(filled_image[signal_mask])
    # mean_signal = np.sum(image * signal_mask) / np.sum(signal_mask)
    implant_mask = ndi.maximum_filter(implant_mask, size=filter_size)
    # image[implant_mask] = mean_signal
    filled_image[implant_mask] = median_signal
    signal_sum = ndi.uniform_filter(filled_image * signal_mask, size=filter_size)
    signal_count = ndi.uniform_filter(signal_mask, size=filter_size, output=float)
    signal_mean =  np.divide(signal_sum, signal_count, out=np.zeros_like(signal_sum), where=signal_count > 0)
    return signal_mean

def get_mask_register(mask_empty, mask_implant, mask_artifact, filter_radius=3):
    mask = (mask_implant + mask_empty + mask_artifact) == 0
    mask = ndi.binary_closing(mask, structure=morphology.ball(filter_radius))
    mask = ndi.binary_opening(mask, structure=morphology.ball(2 * filter_radius))
    return mask

def get_mask_artifact(error, signal_ref):
    mask, _ = get_mask_extrema(error, signal_ref, 0.3, 'mean', abs_margin=True)
    return mask

def get_mask_hyper(error, signal_ref):
    mask, _ = get_mask_extrema(error, signal_ref, 0.3, 'mean', abs_margin=False)
    return mask

def get_mask_hypo(error, signal_ref):
    mask, _ = get_mask_extrema(error, signal_ref, -0.3, 'mean', abs_margin=False)
    return mask

def get_mask_extrema(error, signal_ref, margin, mode, filter_size=5, abs_margin=True):
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
        mask = filtered_error > np.abs(margin) * signal_ref
    else:
        mask = filtered_error * np.sign(margin) > np.abs(margin) * signal_ref
    return mask, filtered_error

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
    artifact = get_mask_artifact(error, signal_ref)

    out = (implant, empty, hyper, hypo, artifact)

    if combine:
        return combine_masks(*out)
    else:
        return out

def print_labels(labels, max_label):
    for i in range(max_label + 1):
        print('label == {} has size {}'.format(i, np.sum(labels==i)))

def combine_masks_2(implant, empty, hyper, hypo, artifact):
    mask = 0 * np.ones(empty.shape)
    mask[artifact] = 3
    mask[hypo] = 2
    mask[hyper] = 4
    mask[empty] = 0
    mask[implant] = 1
    return mask / 4

def combine_masks(implant, empty, hyper, hypo, artifact):
    mask = 2 * np.ones(empty.shape)
    mask[artifact] = 3
    mask[hypo] = 4
    mask[hyper] = 5
    mask[empty] = 0
    mask[implant] = 1
    return mask / 5

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
    return snr, signal, noise

def estimate_psf(clean_image, blurred_image, reg=0.1):
    # https://scikit-image.org/docs/stable/api/skimage.restoration.html#skimage.restoration.wiener
    max_val = np.max(clean_image)
    blurred_image = blurred_image / max_val
    clean_image = clean_image / max_val
    psf = restoration.wiener(blurred_image, clean_image, reg)
    return psf

def estimate_geometric_distortion(fixed_image, moving_image, fixed_mask, moving_mask):
    fixed_image_masked = fixed_image.copy()
    fixed_image_masked[~fixed_mask] = 0
    moving_image_masked = moving_image.copy()
    moving_image_masked[~moving_mask] = 0
    result, transform = register.nonrigid(fixed_image, moving_image, fixed_mask, moving_mask)
    result_masked = register.transform(moving_image_masked, transform)
    deformation_field = register.get_deformation_field(moving_image, transform)
    _, jacobian_det = register.get_jacobian(moving_image, transform)
    return deformation_field, jacobian_det, result, result_masked