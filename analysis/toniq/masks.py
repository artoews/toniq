import numpy as np
import scipy.ndimage as ndi
from skimage import filters, morphology

def get_implant_mask(image):
    mask = image < filters.threshold_otsu(image)  # global Otsu
    mask = morphology.binary_erosion(mask, morphology.ball(3)) # min
    mask = morphology.binary_dilation(mask, morphology.ball(5)) # max
    return mask

def get_artifact_mask(artifact_map, threshold, empty=True):
    if empty:
        artifact_map = ndi.median_filter(np.abs(artifact_map), footprint=morphology.ball(1))
    else:
        # temporary fix for artifact masks with lattice
        artifact_map = np.abs(ndi.generic_filter(artifact_map, np.mean, footprint=morphology.ball(3))) 
    mask = artifact_map > threshold
    mask = morphology.binary_closing(mask, morphology.ball(5)) # closing = dilation (max) then erosion (min)
    return mask

def get_signal_mask(implant_mask, artifact_masks=None):
    """ returns the complement of the closed union of implant & artifact masks """
    if artifact_masks is None:
        artifact_masks = [np.zeros_like(implant_mask)]
    else:
        assert type(artifact_masks) is list
    mask = ((implant_mask + sum(artifact_masks)) > 0)
    mask = morphology.binary_closing(mask, morphology.ball(5))
    return ~mask
