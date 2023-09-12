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

series_dirs_bw31 = [
    '230830/13511_dicom/Series4',
    '230830/13511_dicom/Series12'
]
series_dirs_bw125 = [
    '230830/13511_dicom/Series7',
    '230830/13511_dicom/Series15'
]
series_dirs_msl = [
    '230830/13511_dicom/Series21',
    '230830/13511_dicom/Series17',
]

series_dirs = [root + s for s in series_dirs_bw31]
fixed_files = Path(series_dirs[0]).glob('*MRDC*')
moving_files = Path(series_dirs[1]).glob('*MRDC*')
fixed_image = dicom.load_series(fixed_files).data
moving_image = dicom.load_series(moving_files).data

fixed_image = analysis.normalize(fixed_image)
moving_image = analysis.equalize(moving_image, fixed_image)

mega_mask = analysis.get_all_masks(fixed_image, moving_image, combine=True)
moving_mask = morphology.erosion(mega_mask == 2/5, morphology.ball(2))
signal_ref = analysis.get_typical_level(fixed_image)

slc = (slice(25, 175), slice(50, 200), slice(10, 70)) # ~ 1e6 active pixels
# slc = (slice(75, 175), slice(100, 200), slice(10, 50)) # ~ 1e5 active pixels
# slc = (slice(100, 150), slice(100, 200), slice(20, 60))  # just the problem area
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
error = tuple(2 * (np.abs(vol - fixed_image) / signal_ref[slc] - 0.3) for vol in volumes)
# volumes = volumes + tuple(2*np.abs(vol - fixed_image) for vol in volumes)
volumes = volumes + error
titles = ('fixed', 'moving', 'moving masked', 'result masked')
fig1, ax1 = plotVolumes(volumes, 2, len(volumes) // 2, titles=titles, figsize=(16, 8))

print('{} of {} pixels active ({:.0f}%)'.format(active_count, total_count, active_count / total_count * 100))

plt.show()