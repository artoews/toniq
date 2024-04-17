import numpy as np
from util import coord_mats

def center_of_mass(mass, coords, axis):
    total_mass = np.sum(mass, axis=axis, keepdims=True)
    normalized_mass = np.divide(mass, total_mass, out=np.zeros(mass.shape), where=(total_mass != 0))
    return np.sum(coords * normalized_mass, axis=axis)

def estimate_field(image_xyzb, offsets):
    return center_of_mass(np.abs(image_xyzb), offsets, -1)

# this function was made to compensate for VAT field in field estimation, but I never got it to work
def grad_maps(gz, gx, shape, res):
    x, _, z = coord_mats(shape, res=res, loc=(0.5, 0.5, 0.5), offset=0.5)
    gz_map = gz * z
    gx_map = gx * x
    return gx_map, gz_map  # kHz