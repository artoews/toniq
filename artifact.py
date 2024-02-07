import numpy as np
from skimage import morphology

from filter import nanmean_filter
from util import safe_divide

def get_artifact_map(plastic_image, metal_image, implant_mask, filter_size=3):
    reference = get_signal_reference(plastic_image, implant_mask, filter_size=filter_size)
    error = metal_image - plastic_image
    artifact_map = safe_divide(error, reference)
    return artifact_map

def get_signal_reference(plastic_image, implant_mask, filter_size=3):
    reference = nanmean_filter(plastic_image, ~implant_mask, morphology.cube(filter_size))
    for i in range(plastic_image.shape[2]):
        reference[..., i][implant_mask[..., i]] = np.nanmedian(plastic_image[..., i])
    return reference
