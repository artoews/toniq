"""Masks for restricting image quality analysis.

"""
import numpy as np
import numpy.typing as npt
import scipy.ndimage as ndi
from skimage import filters, morphology

def get_implant_mask(image: npt.NDArray) -> npt.NDArray[np.bool]:
    """Compute a mask including just the implant body.
    
    Note: this implementation is actually the complement to how it is described in the TONIQ paper.
    """
    mask = image < filters.threshold_otsu(image)  # global Otsu
    mask = morphology.binary_erosion(mask, morphology.ball(3)) # min
    mask = morphology.binary_dilation(mask, morphology.ball(5)) # max
    return mask

def get_artifact_mask(
        artifact_map: npt.NDArray[np.float64],
        threshold: float,
        ) -> npt.NDArray[np.bool]:
    """Compute a mask including just the image regions with artifact above a given threshold.
    
    Note: this implementation is actually the complement to how it is described in the TONIQ paper. 
    """
    artifact_map = ndi.median_filter(np.abs(artifact_map), footprint=morphology.ball(1))
    mask = artifact_map > threshold
    mask = morphology.binary_closing(mask, morphology.ball(5))
    return mask

def get_signal_mask(
        implant_mask: npt.NDArray[np.bool],
        artifact_masks: list[npt.NDArray[np.bool]] = None
        ) -> npt.NDArray[np.bool]:
    """ Returns the complement of the (morphologically closed) union of implant & artifact masks. """
    if artifact_masks is None:
        artifact_masks = [np.zeros_like(implant_mask)]
    else:
        assert type(artifact_masks) is list
    mask = ((implant_mask + sum(artifact_masks)) > 0)
    mask = morphology.binary_closing(mask, morphology.ball(5))
    return ~mask
