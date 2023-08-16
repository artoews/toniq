import numpy as np
from scipy.ndimage import median_filter

def signal_void(artifact_image, reference_image=None, threshold=0.7, filter_size=10):
    if reference_image is None:
        reference_image = np.ones_like(artifact_image) * np.mean(artifact_image)
    mask = artifact_image < reference_image * threshold
    mask = median_filter(mask, size=filter_size)
    return mask
