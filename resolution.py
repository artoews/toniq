import numpy as np

import analysis
from psf import estimate_psf
from fwhm import get_FWHM_from_image
from util import safe_divide

def get_mask(reference, target, metal=False):
    mask_empty = analysis.get_mask_empty(reference)
    mask_implant = analysis.get_mask_implant(mask_empty)
    if metal:
        mask_signal = analysis.get_mask_signal(reference)
        signal_ref = analysis.get_typical_level(reference, mask_signal, mask_implant)
        error = target - reference 
        normalized_error = safe_divide(error, signal_ref)
        mask_artifact = analysis.get_mask_artifact(normalized_error)
        mask = analysis.get_mask_register(mask_empty, mask_implant, mask_artifact)
    else:
        mask_lattice = analysis.get_mask_lattice(reference)
        mask = np.logical_and(mask_lattice, ~mask_implant)
    return mask

def map_resolution(reference, target, unit_cell, stride=8, num_workers=8, mask=None):
    num_dims = target.ndim
    patch_shape = (unit_cell,) * num_dims # might want to double for better noise robustness
    if mask is None:
        mask = get_mask(reference, target)
    psf = estimate_psf(reference, target, mask, patch_shape, stride, num_workers)
    fwhm = get_FWHM_from_image(psf, num_workers)
    return psf, fwhm
