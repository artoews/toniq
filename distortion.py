import numpy as np
import scipy.ndimage as ndi

from os import path
from skimage import morphology

def net_pixel_bandwidth(pixel_bandwidth_2, pixel_bandwidth_1):
    if pixel_bandwidth_1 == 0:
        return pixel_bandwidth_2
    else:
        return 1 / (1 / pixel_bandwidth_2 - 1 / pixel_bandwidth_1)

def get_true_field(field_dir):
    metal_field = np.load(path.join(field_dir, 'field-metal.npy')) # kHz
    plastic_field = np.load(path.join(field_dir, 'field-plastic.npy'))  # kHz
    true_field = metal_field - plastic_field
    true_field = ndi.median_filter(true_field, footprint=morphology.ball(4))
    # true_field = ndi.generic_filter(true_field, np.mean, footprint=morphology.ball(3))
    return true_field



