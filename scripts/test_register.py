import matplotlib.pyplot as plt
import numpy as np

import analysis
import dicom
import register

fpaths = [
    '/bmrNAS/people/artoews/data/scans/230713/13160_dicom/Series3/i43556305.MRDC.13',
    '/bmrNAS/people/artoews/data/scans/230713/13160_dicom/Series6/i43556413.MRDC.13'
]

images = [dicom.read_data(f)[70:190, 80:170] for f in fpaths]
fixed_image = images[1]
moving_image = images[0]

moving_mask = np.logical_not(analysis.mask_signal_void(moving_image, fixed_image))
moving_image_masked = moving_image.copy()
moving_image_masked[~moving_mask] = 0

# Plot images
fig, axs = plt.subplots(1, 3, sharey=True, figsize=[30,30])
plt.figsize=[100,100]
axs[0].imshow(moving_image)
axs[1].imshow(moving_mask)
axs[2].imshow(moving_image_masked)
plt.show()

result, transform = register.nonrigid(fixed_image, moving_image, moving_mask)

# Plot images
fig, axs = plt.subplots(1,4, sharey=True, figsize=[30,30])
plt.figsize=[100,100]
axs[0].imshow(fixed_image)
axs[0].set_title('Fixed', fontsize=30)
axs[1].imshow(moving_image)
axs[1].set_title('Moving', fontsize=30)
axs[2].imshow(moving_image_masked)
axs[2].set_title('Moving Masked', fontsize=30)
mask = np.asarray(moving_mask, dtype=bool)
result_masked = np.zeros_like(result)
result_masked[mask] = np.asarray(result)[mask]
axs[3].imshow(result_masked)
axs[3].set_title('Result Masked', fontsize=30)
plt.show()