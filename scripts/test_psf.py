import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import scipy.ndimage as ndi

import analysis
import dicom
import psf
from plot import plotVolumes

# TODO plot the PSF kernel as a 3D object you can rotate around, plot alongside the image patches
# TODO come up with a plotting solution for showing the PSF kernel and image patches with different shapes. What kind of scrolling do I want?
# TODO experiment with different sizes of the image patch and kernel to settle on some reasonable values in terms of time and resolution
# TODO setup efficient solving over many overlapping patches, e.g. solve disparate patches in parallel, then use those results to initialize neighbors

root = '/Users/artoews/root/data/mri/'

series_dirs_msl = [
    '230830/13511_dicom/Series5',
    '230830/13511_dicom/Series6',
]

series_dirs = [root + s for s in series_dirs_msl]
image1_files = Path(series_dirs[0]).glob('*MRDC*')
image2_files = Path(series_dirs[1]).glob('*MRDC*')
image1 = dicom.load_series(image1_files)
image2 = dicom.load_series(image2_files)

image1.data = analysis.normalize(image1.data)
image2.data = analysis.equalize(image2.data, image1.data)

mask_empty = analysis.get_mask_empty(image1.data)
mask_implant = analysis.get_mask_implant(mask_empty)
mask_signal = analysis.get_mask_signal(image1.data, image2.data)

error = image1.data - image2.data
# psf = analysis.estimate_psf(image2.data, image1.data, reg=100)

def upsample(psf, size):
    factor = size / psf.shape[0]
    return np.repeat(np.repeat(np.repeat(psf, factor, axis=0), factor, axis=1), factor, axis=2)

patch_size = 24
psf_size = 6
l2 = 1e-5
image_ref = image2.data
image_blur = image1.data
image_blur = ndi.gaussian_filter(image_blur, 1, axes=1)
image_ref_patch, image_blur_patch, psf_init, psf_soln = psf.estimate_psf(image_ref, image_blur, patch_size, psf_size)

psf_init = upsample(np.abs(psf_init), patch_size + psf_size)
psf_soln = upsample(np.abs(psf_soln), patch_size + psf_size)

psf_init = psf_init / np.max(psf_soln)
psf_soln = psf_soln / np.max(psf_soln)

factor = 1
volumes = (image_ref_patch, image_blur_patch, psf_init * factor, psf_soln * factor)
titles = ('Clean', 'Blurred', 'Init', 'Soln')
fig1, tracker1 = plotVolumes(volumes, 1, len(volumes), titles=titles, figsize=(16, 8))

# factor = 5
# volumes = (image1.data, image2.data, (error * factor + 1) / 2, np.abs(error) * factor, psf * factor)
# titles = ('Image 1', 'Image 2', 'error', 'abs error', 'PSF')
# fig2, tracker2 = plotVolumes(volumes, 1, len(volumes), titles=titles, figsize=(16, 8))

plt.show()