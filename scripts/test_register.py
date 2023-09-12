import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from skimage import morphology

import analysis
import dicom
import register
from plot import plotVolumes

root = '/Users/artoews/root/data/mri/'
# root = '/bmrNAS/people/artoews/data/scans/'

series_dirs_old = [
    '230713/13160_dicom/Series6',
    '230713/13160_dicom/Series3'
]
series_dirs_bw31 = [
    '230830/13511_dicom/Series4',
    '230830/13511_dicom/Series12'
]

series_dirs = [root + s for s in series_dirs_bw31]
fixed_files = Path(series_dirs[0]).glob('*MRDC*')
moving_files = Path(series_dirs[1]).glob('*MRDC*')
fixed_image = dicom.load_series(fixed_files).data
moving_image = dicom.load_series(moving_files).data

fixed_image = analysis.normalize(fixed_image)
moving_image = analysis.equalize(moving_image, fixed_image)
signal_ref = analysis.get_typical_level(fixed_image)
error = moving_image - fixed_image
denoised_error = analysis.denoise(error)
signal_ref = analysis.get_typical_level(fixed_image)
mask_empty = analysis.get_mask_empty(fixed_image)
mask_implant = analysis.get_mask_implant(mask_empty)
mask_hyper = analysis.get_mask_hyper(denoised_error, signal_ref)
mask_hypo = analysis.get_mask_hypo(denoised_error, signal_ref)
mask_artifact = analysis.get_mask_artifact(denoised_error, signal_ref)

# TODO make a function that just generates all the masks for you
# TODO debug why this mask looks different than the one I'm getting with demo_map_combo
moving_mask = np.logical_not(np.logical_or(np.logical_or(np.logical_or(np.logical_or(mask_hyper, mask_hypo), mask_artifact), mask_empty), mask_implant))
moving_mask = morphology.dilation(moving_mask, morphology.ball(2))

# slc = (slice(25, 175), slice(50, 200), slice(10, 70)) # ~ 1e6 active pixels
slc = (slice(75, 175), slice(100, 200), slice(20, 60)) # ~ 1e5 active pixels
# slc = (slice(20, 60), slice(100, 140), slice(30, 50)) # ~ 1e4 active pixels?
fixed_image = fixed_image[slc]
moving_image = moving_image[slc]
moving_mask = moving_mask[slc]

active_count = np.sum(moving_mask)
total_count = np.prod(moving_mask.shape)
print('{} of {} pixels active ({:.0f}%)'.format(active_count, total_count, active_count / total_count * 100))

moving_image_masked = moving_image.copy()
moving_image_masked[~moving_mask] = 0
# fixed_image_masked = fixed_image.copy()
# fixed_image_masked[~moving_mask] = 0  # TODO ideally you would warp the moving_mask to create a fixed mask

# Plot images
volumes = (moving_image, moving_mask, moving_image_masked)
titles = ('moving image', 'reg mask', 'moving image (masked)')
# fig0, ax0 = plotVolumes(volumes, 1, len(volumes), titles=titles, figsize=(16, 8))

result, transform = register.nonrigid(fixed_image, moving_image, moving_mask)

# Plot images
result_masked = np.zeros_like(result)
mask = np.asarray(moving_mask, dtype=bool)
result_masked[mask] = np.asarray(result)[mask]
volumes = (fixed_image, moving_image, moving_image_masked, result_masked)
volumes = volumes + tuple(2*np.abs(vol - fixed_image) for vol in volumes)
titles = ('fixed', 'moving', 'moving masked', 'result masked')
fig1, ax1 = plotVolumes(volumes, 2, len(volumes) // 2, titles=titles, figsize=(16, 8))

print('{} of {} pixels active ({:.0f}%)'.format(active_count, total_count, active_count / total_count * 100))

plt.show()