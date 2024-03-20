import matplotlib.pyplot as plt
import numpy as np
import sigpy as sp
import functools

from filter import generic_filter
from linop import get_matrix

from plot import overlay_mask, colorbar_axis
from plot_params import *


def get_map(reference, target, psf_shape, patch_shape, resolution_mm, mask, stride, num_workers=1):
    psf = estimate_psf(reference, target, mask, psf_shape, patch_shape, stride, num_workers)
    fwhm = get_FWHM_from_image(psf, psf_shape, num_workers)
    for i in range(fwhm.shape[-1]):
        fwhm[..., i] = fwhm[..., i] * resolution_mm[i]
    if stride == 1:
        psf = sp.resize(psf, target.shape[:2] + psf.shape[2:])
        fwhm = sp.resize(fwhm, target.shape[:2] + fwhm.shape[2:])
    return psf, fwhm

def estimate_psf(image_in, image_out, mask, psf_shape, patch_shape, stride, num_batches):
    images_stack = np.stack((image_in, image_out), axis=-1)
    images_stack[~mask, ...] = np.nan
    batch_axis = 2
    func = functools.partial(deconvolve_by_model, psf_shape)
    return generic_filter(images_stack, func, patch_shape, psf_shape, stride, batch_axis, num_batches=num_batches)

def deconvolve_by_model(psf_shape, patch_pair, verbose=False):
    # kspace_pair = sp.fft(patch_pair, axes=(0, 1))
    # kspace_in, kspace_out = kspace_pair[..., 0], kspace_pair[..., 1]
    # A = forward_model(kspace_in, psf_shape)
    A = forward_model_conv(patch_pair[..., 0], psf_shape)
    y = sp.resize(patch_pair[..., 1], A.oshape)
    # A_mat = get_matrix(A)
    # A_mat = np.random.rand(np.prod(A.oshape), np.prod(A.ishape))
    # A_inv = np.linalg.pinv(A_mat) # direct solve takes about 4 times longer than iterative solve using a random matrix as a proxy for A. If you call get_matrix and actually construct A by brute force it takes 10x instead of 4x.
    # soln = np.reshape(A_inv.dot(y.ravel()), psf_shape)
    app = sp.app.LinearLeastSquares(A, y, x=np.zeros(A.ishape, dtype=np.float64), tol=1e-10, max_iter=1e10, show_pbar=verbose, lamda=0)
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

def forward_model_conv(input_image, psf_shape):
    A = sp.linop.ConvolveFilter(psf_shape, input_image, mode='valid')
    return A

def get_FWHM_from_image(psf, psf_shape, num_workers, stride=1, batch_axis=2):
    func = get_FWHM_from_pixel
    patch_shape = (1, 1, 1)
    out_shape = (sum(p > 1 for p in psf_shape),) # ndim of psf
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

def plot_map(ax, res_map, mask, vmin=1, vmax=4, show_cbar=True):
    im = ax.imshow(res_map, cmap=CMAP['resolution'], vmin=vmin, vmax=vmax)
    if mask is not None:
        overlay_mask(ax, ~mask)
    if show_cbar:
        cbar = colorbar(ax, im, 'FWHM (mm)', ticks=[vmin, vmin + (vmax-vmin)/2, vmax])
        # cbar = plt.colorbar(im, cax=colorbar_axis(ax), ticks=[vmin, vmin + (vmax-vmin)/2, vmax])
        return cbar

def colorbar(ax, im, label, offset=0, ticks=[1, 2, 3]):
    cbar = plt.colorbar(im, cax=colorbar_axis(ax, offset=offset), ticks=ticks)
    cbar.set_label(label, size=SMALL_SIZE)
    cbar.ax.tick_params(labelsize=SMALLER_SIZE)
    return cbar
