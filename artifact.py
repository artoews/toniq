import scipy.ndimage as ndi
from skimage import morphology

from util import safe_divide

def get_artifact_map(plastic_image, metal_image, implant_mask, filter_size=3):
    error = metal_image - plastic_image
    reference = ndi.median_filter(plastic_image, footprint=morphology.cube(filter_size))
    artifact_map = safe_divide(error, reference)
    artifact_map[implant_mask] = 0
    return artifact_map
