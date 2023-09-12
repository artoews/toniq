import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import scipy.ndimage as ndi
import sigpy as sp

import analysis
import dicom
import psf
from plot import plotVolumes

# TODO see what time saving can be gained from a good initialization. Is it 10x? 100x?
# TODO setup efficient solving over many overlapping patches, e.g. solve disparate patches in parallel, then use those results to initialize neighbors

root = '/Users/artoews/root/data/mri/'

series_dirs_msl_small = [
    '230830/13511_dicom/Series5',
    '230830/13511_dicom/Series6'
]

series_dirs_msl_large = [
    '230817/13427_dicom/Series2',
    '230817/13427_dicom/Series5'
]

series_dirs = [root + s for s in series_dirs_msl_large]
image1_files = Path(series_dirs[0]).glob('*MRDC*')
image2_files = Path(series_dirs[1]).glob('*MRDC*')
image1 = dicom.load_series(image1_files)
image2 = dicom.load_series(image2_files)

image1.data = analysis.normalize(image1.data)
image2.data = analysis.equalize(image2.data, image1.data)

mask_empty = analysis.get_mask_empty(image1.data)
mask_implant = analysis.get_mask_implant(mask_empty)

error = image1.data - image2.data
# psf = analysis.estimate_psf(image2.data, image1.data, reg=100)

def upsample(psf, size):
    factor = size / psf.shape[0]
    return np.repeat(np.repeat(np.repeat(psf, factor, axis=2), factor, axis=1), factor, axis=2)

def impulse(shape):
    return sp.resize(np.ones((1,) * len(shape)), shape)

# cell_size = 10
cell_size = 14
psf_size = 7
patch_size = cell_size * 2
image_ref = image1.data
image_blur = image2.data
blur_sigma = 0.8
blur_axis = 1
# image_blur = ndi.gaussian_filter(image_ref, blur_sigma, axes=blur_axis)
imp = impulse((psf_size,) * image_blur.ndim)
psf_true = ndi.gaussian_filter(imp, blur_sigma, axes=blur_axis)
image_ref_patch, image_blur_patch, psf_init, psf_est = psf.estimate_psf(image_ref, image_blur, patch_size, psf_size)

interp_factor = 8
psf_int = psf.interpolate_sinc(psf_est, psf_size * interp_factor)

psf_init = np.abs(psf_init)
psf_est = np.abs(psf_est)
psf_int = np.abs(psf_int)

norm = np.max(psf_est)
psf_init = psf_init / norm 
psf_soln = psf_est / norm
psf_int = psf_int / norm
psf_true = psf_true / np.max(psf_true)

max_idx = np.unravel_index(np.argmax(psf_int), psf_int.shape)
fwhm_x = psf.get_FWHM(psf_int[:, max_idx[1], max_idx[2]]) / interp_factor
fwhm_y = psf.get_FWHM(psf_int[max_idx[0], :, max_idx[2]]) / interp_factor
fwhm_z = psf.get_FWHM(psf_int[max_idx[0], max_idx[1], :]) / interp_factor
print('FWHM', fwhm_x, fwhm_y, fwhm_z)

volumes = (image_ref, image_blur)
titles = ('Clean', 'Blurred')
fig0, tracker0 = plotVolumes(volumes, 1, len(volumes), titles=titles, figsize=(16, 8))

volumes = (image_ref_patch, image_blur_patch)
titles = ('Clean', 'Blurred')
fig1, tracker1 = plotVolumes(volumes, 1, len(volumes), titles=titles, figsize=(16, 8))

volumes = (imp, psf_true, psf_soln)
titles = ('Impulse', 'True PSF', 'Estimated PSF')
fig2, tracker2 = plotVolumes(volumes, 1, len(volumes), titles=titles, figsize=(16, 8))

fwhm_string = '{:.1f}, {:.1f}, {:.1f} pixels'.format(fwhm_x, fwhm_y, fwhm_z)
print(fwhm_string)
fig3, tracker3 = plotVolumes((np.zeros_like(psf_int), psf_int), 1, 2, titles=('Zero', '{}x Sinc-Interpolated PSF with FWHM: '.format(interp_factor) + fwhm_string), figsize=(16, 8))

plt.show()