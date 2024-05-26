import numpy as np

def center_of_mass(mass, coords, axis):
    total_mass = np.sum(mass, axis=axis, keepdims=True)
    normalized_mass = np.divide(mass, total_mass, out=np.zeros(mass.shape), where=(total_mass != 0))
    return np.sum(coords * normalized_mass, axis=axis)

def estimate_field(image_xyzb, offsets):
    return center_of_mass(np.abs(image_xyzb), offsets, -1)
