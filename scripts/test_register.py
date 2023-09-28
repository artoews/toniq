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

series_dirs = [root + s for s in series_dirs_msl]
fixed_files = Path(series_dirs[0]).glob('*MRDC*')
moving_files = Path(series_dirs[1]).glob('*MRDC*')
fixed_image = dicom.load_series(fixed_files).data
moving_image = dicom.load_series(moving_files).data

fixed_image = analysis.normalize(fixed_image)
moving_image = analysis.equalize(moving_image, fixed_image)

mega_mask = analysis.get_all_masks(fixed_image, moving_image, combine=True)
fixed_mask = np.logical_not(analysis.get_mask_empty(fixed_image))
fixed_mask = morphology.erosion(fixed_mask, morphology.ball(2))
moving_mask = (mega_mask == 2/5)
moving_mask = morphology.erosion(moving_mask, morphology.ball(2))
signal_ref = analysis.get_typical_level(fixed_image)

# slc = (slice(None), slice(None), slice(None))
slc = (slice(25, 175), slice(50, 200), slice(10, 70)) # ~1e6 active pixels - All of them
# slc = (slice(75, 175), slice(100, 200), slice(10, 50)) # ~2e5 active pixels - 20% of them
# slc = (slice(100, 150), slice(100, 200), slice(20, 60))  # just the problem area
fixed_image = fixed_image[slc]
fixed_mask = fixed_mask[slc]
moving_image = moving_image[slc]
moving_mask = moving_mask[slc]

active_count = np.sum(moving_mask)
total_count = np.prod(moving_mask.shape)
print('{} of {} pixels active ({:.0f}%)'.format(active_count, total_count, active_count / total_count * 100))

fixed_image_masked = fixed_image.copy()
fixed_image_masked[~fixed_mask] = 0
moving_image_masked = moving_image.copy()
moving_image_masked[~moving_mask] = 0

result, transform = register.nonrigid(fixed_image, moving_image, fixed_mask, moving_mask)
result_masked = register.transform(moving_image_masked, transform)

deformation_field = register.get_deformation_field(moving_image, transform)
_, jacobian_det = register.get_jacobian(moving_image, transform)

np.save('deformation_field_msl.npy', deformation_field)

deformation_field_x = deformation_field[..., 0] / np.max(np.abs(deformation_field[..., 0])) / 2 + 0.5
deformation_field_y = deformation_field[..., 1] / np.max(np.abs(deformation_field[..., 1])) / 2 + 0.5
deformation_field_z = -deformation_field[..., 2] / np.max(np.abs(deformation_field[..., 2])) / 2 + 0.5
jacobian_det = jacobian_det / np.max(np.abs(jacobian_det)) / 2 + 0.5
# TODO deform the moving mask before using it here
deformation_field_x[~moving_mask] = 0
deformation_field_y[~moving_mask] = 0
deformation_field_z[~moving_mask] = 0
jacobian_det[~fixed_mask] = 0

# Plot images

volumes = (fixed_image, moving_image, result, deformation_field_x, deformation_field_y, fixed_image_masked, moving_image_masked, result_masked, deformation_field_z, jacobian_det)
titles = ('fixed', 'moving', 'result', 'deformation x', 'deformation y', 'fixed masked', 'moving masked', 'result masked', 'deformation z', 'jacobian det')
fig0, ax0 = plotVolumes(volumes, 2, len(volumes) // 2, titles=titles, figsize=(16, 8))

volumes = (fixed_image, moving_image, result, fixed_image_masked, moving_image_masked, result_masked)
error = tuple(2 * (np.abs(vol - fixed_image) / signal_ref[slc] - 0.3) for vol in volumes)
volumes = volumes + error
titles = ('fixed', 'moving', 'result', 'fixed masked', 'moving masked', 'result masked')
fig1, ax1 = plotVolumes(volumes, 2, len(volumes) // 2, titles=titles, figsize=(16, 8))

print('{} of {} pixels active ({:.0f}%)'.format(active_count, total_count, active_count / total_count * 100))

plt.show()