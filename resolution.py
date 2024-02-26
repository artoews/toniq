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
    # patch_shape = (unit_cell_pixels[0], unit_cell_pixels[0], unit_cell_pixels[0])
    filter_size = (int(patch_shape[0]/stride/2), int(patch_shape[1]/stride/2), int(patch_shape[2]/2))
    print('patch shape', patch_shape)
    # print('filter size', filter_size)
    psf = estimate_psf(reference, target, mask, patch_shape, stride, num_workers)
    fwhm = get_FWHM_from_image(psf, num_workers)
    for i in range(fwhm.shape[-1]):
        fwhm[..., i] = fwhm[..., i] * resolution_mm[i]
        # fwhm[..., i] = ndi.uniform_filter(fwhm[..., i], size=filter_size)
        # fwhm[..., i] = ndi.median_filter(fwhm[..., i], footprint=np.ones(filter_size))
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

def deconvolve_by_model(patch_pair, lamda=1e-3, tol=1e-8, max_iter=1e4, verbose=False):
    # kspace_pair = sp.fft(patch_pair, axes=(0, 1, 2))
    kspace_pair = sp.fft(patch_pair, axes=(0, 1))
    kspace_in, kspace_out = kspace_pair[..., 0], kspace_pair[..., 1]
    # A = forward_model(kspace_in, psf_shape)
    # y = kspace_out
    # A = forward_model_2(kspace_in, psf_shape)
    # y = sp.resize(sp.ifft(kspace_out), A.oshape)
    A = forward_model_22(kspace_in, psf_shape)
    # y = kspace_out
    # A = forward_model_k(kspace_in, psf_shape)
    # A = forward_model_diff(kspace_in, psf_shape, lamda)
    y = sp.resize(sp.ifft(kspace_out, axes=(0, 1)), A.oshape)
    # y = np.concatenate((y.ravel(), np.zeros(np.prod(A.oshape) - y.size)))
    # y = np.concatenate((kspace_out.ravel(), np.zeros(np.prod(A.oshape) - kspace_out.size)))
    # print(A)
    # print(y.shape)
    app = sp.app.LinearLeastSquares(A, y, x=np.zeros(A.ishape, dtype=np.complex128), tol=tol, max_iter=max_iter, show_pbar=verbose, lamda=lamda)
    soln = app.run()
    # psf = np.abs(sp.ifft(sp.resize(soln, kspace_in.shape[:2] + psf_shape[2:])))
    psf = np.abs(soln)
    return psf

def forward_model(input_kspace, psf_shape):
    Z = sp.linop.Resize(input_kspace.shape, psf_shape)
    F = sp.linop.FFT(Z.oshape)
    D = sp.linop.Multiply(F.oshape, input_kspace)
    return D * F * Z

def forward_model_2(input_kspace, psf_shape):
    # expects 3D k-space and 3D psf with singleton 3rd dimension
    Z = sp.linop.Resize(input_kspace.shape, psf_shape)
    F = sp.linop.FFT(Z.oshape)
    D = sp.linop.Multiply(F.oshape, input_kspace)
    no_wrap_size = tuple(np.array(input_kspace.shape[:2]) - np.array(psf_shape[:2]) + np.ones(2, dtype=int))
    C = sp.linop.Resize(no_wrap_size + input_kspace.shape[2:], F.ishape)
    # return D * F * Z
    return C * F.H * D * F * Z

def forward_model_22(input_kspace, psf_shape):
    # expects stack of 2D k-space slices and 3D psf with singleton 3rd dimension
    Z = sp.linop.Resize(input_kspace.shape[:2] + (1,), psf_shape)
    F = sp.linop.FFT(Z.oshape, axes=(0, 1))
    D = sp.linop.Multiply(F.oshape, input_kspace)
    no_wrap_size = tuple(np.array(input_kspace.shape[:2]) - np.array(psf_shape[:2]) + np.ones(2, dtype=int))
    FH = sp.linop.FFT(D.oshape, axes=(0, 1)).H
    C = sp.linop.Resize(no_wrap_size + input_kspace.shape[2:], FH.oshape)
    return C * FH * D * F * Z
    # return D * F * Z

def forward_model_diff(input_kspace, psf_shape, lamda):
    A = forward_model_22(input_kspace, psf_shape)
    D = np.sqrt(lamda) * sp.linop.FiniteDifference(psf_shape)
    return sp.linop.Vstack((A, D))

# def forward_model_k(input_kspace, psf_shape):
#     # expects stack of 2D k-space slices and 3D psf with singleton 3rd dimension
#     kspace_shape = tuple(s // 2 for s in input_kspace.shape[:2]) + input_kspace.shape[2:]
#     active_kspace = sp.resize(input_kspace, kspace_shape)
#     Z1 = sp.linop.Resize(input_kspace.shape[:2] + (1,), psf_shape)
#     F = sp.linop.FFT(Z1.oshape, axes=(0, 1))
#     Z2 = sp.linop.Resize(active_kspace.shape[:2] + (1,), F.oshape)
#     D = sp.linop.Multiply(Z2.oshape, active_kspace)
#     return D * Z2 * F * Z1

def forward_model_explicit(kspace, psf_shape):
    # old, and not sure this was ever proved to be correct
    shape = kspace.shape
    pad_mat = np.diag(sp.resize(np.ones(psf_shape), kspace.shape).ravel())
    dft_mat = np.kron(dft(shape[0]), np.kron(dft(shape[1]), dft(shape[2])))
    k_mat = np.diag(kspace.ravel())
    mat = k_mat @ dft_mat @ pad_mat
    return mat

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
