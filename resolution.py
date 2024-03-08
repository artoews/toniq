import numpy as np
import scipy.ndimage as ndi
import sigpy as sp
from scipy.linalg import dft

import masks
from filter import generic_filter
from util import safe_divide

psf_shape = (5, 5, 1)
psf_ndim = 2

def map_resolution(reference, target, patch_shape, resolution_mm, mask, stride, num_workers=1):
    psf = estimate_psf(reference, target, mask, patch_shape, stride, num_workers)
    fwhm = get_FWHM_from_image(psf, num_workers)
    for i in range(fwhm.shape[-1]):
        fwhm[..., i] = fwhm[..., i] * resolution_mm[i]
    # psf = sp.resize(psf, (psf.shape[0] + patch_shape[0], psf.shape[1] + patch_shape[1],) + psf.shape[2:])
    # fwhm = sp.resize(fwhm, (fwhm.shape[0] + patch_shape[0], fwhm.shape[1] + patch_shape[1],) + fwhm.shape[2:])
    psf = sp.resize(psf, target.shape[:2] + psf.shape[2:])
    fwhm = sp.resize(fwhm, target.shape[:2] + fwhm.shape[2:])
    return psf, fwhm

def estimate_psf(image_in, image_out, mask, patch_shape, stride, num_batches):
    images_stack = np.stack((image_in, image_out), axis=-1)
    images_stack[~mask, ...] = np.nan
    batch_axis = 2
    func = deconvolve_by_model
    return generic_filter(images_stack, func, patch_shape, psf_shape, stride, batch_axis, num_batches=num_batches)

def deconvolve_by_division(patch_pair):
    kspace_pair = sp.fft(patch_pair, axes=(0, 1, 2))
    kspace_in, kspace_out = kspace_pair[..., 0], kspace_pair[..., 1]
    kspace_quotient = safe_divide(kspace_out, kspace_in, thresh=1e-3)  # TODO set thresh adaptively, say based on noise std?
    psf = np.real(sp.ifft(kspace_quotient))
    return psf

def deconvolve_by_model(patch_pair, lamda=0, tol=1e-10, max_iter=1e10, verbose=False):
    kspace_pair = sp.fft(patch_pair, axes=(0, 1))
    kspace_in, kspace_out = kspace_pair[..., 0], kspace_pair[..., 1]
    A = forward_model(kspace_in, psf_shape)
    y = sp.resize(patch_pair[..., 1], A.oshape)
    # A = forward_model_conv(patch_pair[..., 0], psf_shape)
    # y = sp.resize(patch_pair[..., 1], A.oshape)
    # print(A)
    app = sp.app.LinearLeastSquares(A, y, x=np.zeros(A.ishape, dtype=np.complex128), tol=tol, max_iter=max_iter, show_pbar=verbose, lamda=lamda)
    soln = app.run()
    psf = np.abs(soln)
    return psf

def forward_model(input_kspace, psf_shape):
    # expects stack of 2D k-space slices and 3D psf with singleton 3rd dimension
    Z = sp.linop.Resize(input_kspace.shape[:2] + (1,), psf_shape)
    F = sp.linop.FFT(Z.oshape, axes=(0, 1))
    D = sp.linop.Multiply(F.oshape, input_kspace)
    no_wrap_size = tuple(np.array(input_kspace.shape[:2]) - np.array(psf_shape[:2]) + np.ones(2, dtype=int))
    FH = sp.linop.FFT(D.oshape, axes=(0, 1)).H
    C = sp.linop.Resize(no_wrap_size + input_kspace.shape[2:], FH.oshape)
    return C * FH * D * F * Z
    # return FH * D * F * Z

def forward_model_conv(input_image, psf_shape):
    A = sp.linop.ConvolveFilter(psf_shape, input_image, mode='valid')
    return A

def get_FWHM_from_image(psf, num_workers, stride=1, batch_axis=2):
    func = get_FWHM_from_pixel
    patch_shape = (1, 1, 1)
    out_shape = (psf_ndim,)
    return generic_filter(psf, func, patch_shape, out_shape, stride, batch_axis, num_batches=num_workers)

def get_FWHM_from_pixel(psf):
    psf = np.abs(np.squeeze(psf))
    ndim = psf.ndim
    i_max = np.unravel_index(np.argmax(psf), psf.shape)
    if psf[i_max] == 0:
        return (0,) * ndim
    fwhm_list = []
    for i in range(ndim):
        slc = list(i_max)
        slc[i] = slice(None)
        slc = tuple(slc)
        fwhm_i = get_FWHM_from_vec(psf[slc], i_max=i_max[i])
        fwhm_list.append(fwhm_i)
    return fwhm_list

def get_FWHM_from_vec(x, i_max=None):
    if i_max is None:
        i_max = np.argmax(x)
    half_max = x[i_max] / 2
    if i_max < 1 or i_max >= len(x) - 1:
        return 0
    i_half_max = i_max - np.argmin(x[i_max::-1] > half_max)  # just left of half-max
    if i_half_max == i_max:
        i_half_max_1 = None
    else:
        i_half_max_1 = find_root(i_half_max,
                                 x[i_half_max] - half_max,
                                 i_half_max + 1,
                                 x[i_half_max + 1] - half_max)
    i_half_max = i_max + np.argmin(x[i_max::1] > half_max)  # just right of half-max
    if i_half_max == i_max:
        i_half_max_2 = None
    else:
        i_half_max_2 = find_root(i_half_max - 1,
                                 x[i_half_max - 1] - half_max,
                                 i_half_max,
                                 x[i_half_max] - half_max)
    if i_half_max_1 is None or i_half_max_2 is None:
        print('Warning: PSF half-max extent exceeds window')
    if i_half_max_1 is not None and i_half_max_2 is not None:
        return i_half_max_2 - i_half_max_1
    if i_half_max_1 is None and i_half_max_2 is not None:
        return 2 * (i_half_max_2 - i_max)
    if i_half_max_1 is not None and i_half_max_2 is None:
        return 2 * (i_max - i_half_max_1)
    if i_half_max_1 is None and i_half_max_2 is None:
        return 0

def find_root(x1, y1, x2, y2):
    # find zero-crossing of line through points (x1, y1) and (x2, y2)
    # TODO add some epsilon for when points are too close together, just return one of them
    if y1 == y2:
        print('find_root division by zero')
    return x1 - y1 * (x2 - x1) / (y2 - y1)

def sinc_fwhm(in_shape, out_shape):
    k = sp.resize(np.ones(out_shape), in_shape)
    psf = sp.ifft(k)
    return get_FWHM_from_pixel(psf)
