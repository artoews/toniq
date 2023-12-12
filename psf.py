import analysis
import itertools
import numpy as np
from multiprocessing import Pool
import sigpy as sp

from util import safe_divide

reg_psf_shape = (7, 7, 7)

def batch_for_filter(arr, filter_size, axis, num_batches):
    """ split array into batches for parallelized filtering """
    starts = np.arange(0, arr.shape[axis] - filter_size + 1)
    if len(starts) < num_batches:
        print('Warning: batching into {} sections cannot be done with only {} positions. Returning {} batches instead.'.format(num_batches, len(starts), len(starts)))
        num_batches = len(starts)
    batches = []
    for batch_starts in np.array_split(starts, num_batches):
        batch_indices = np.arange(batch_starts[0], batch_starts[-1] + filter_size)
        batch_data = np.take(arr, tuple(batch_indices), axis=axis)
        batches.append(batch_data)
    return batches

def generic_filter(image, function, filter_shape, out_shape, stride, batch_axis, num_batches=1):
    ''' strided filter accepting generic function (for both PSF and FWHM estimation) '''
    if num_batches > 1:
        image_batches = batch_for_filter(image, filter_shape[batch_axis], batch_axis, num_batches)
        num_batches = len(image_batches)
        inputs = list(zip(
            image_batches,
            (function,) * num_batches,
            (filter_shape,) * num_batches,
            (out_shape,) * num_batches,
            (stride,) * num_batches,
            (batch_axis,) * num_batches,
            ))
        with Pool(num_batches) as p:
            result = p.starmap(generic_filter, inputs)
        return np.concatenate(result, axis=batch_axis)
    else:
        strides = np.roll((1, stride, stride), batch_axis)  # only stride non-batch axes
        patch_locs = tuple(np.arange(0, image.shape[i] - filter_shape[i] + 1, strides[i]) for i in range(3))
        output = np.zeros(tuple(len(locs) for locs in patch_locs) + out_shape)
        for loc in itertools.product(*patch_locs):
            slc = tuple(slice(loc[i], loc[i] + filter_shape[i]) for i in range(3))
            idx = tuple(loc[i] // strides[i] for i in range(3))
            patch = image[slc]
            if np.isnan(patch).any():
                output[idx] = np.zeros(out_shape)
            else:
                output[idx] = function(patch)
        return output

def estimate_psf(image_in, image_out, mask, patch_shape, stride, num_batches, mode='regularized'):
    images_stack = np.stack((image_in, image_out), axis=-1)
    images_stack[~mask, ...] = np.nan
    batch_axis = 2
    if mode == 'division':
        func = estimate_psf_kspace
        psf_shape = patch_shape
    elif mode == 'regularized':
        func = estimate_psf_reg
        psf_shape = reg_psf_shape
    return generic_filter(images_stack, func, patch_shape, psf_shape, stride, batch_axis, num_batches=num_batches)

def estimate_psf_kspace(patch_pair):
    kspace_pair = sp.fft(patch_pair, axes=(0, 1, 2))
    kspace_in, kspace_out = kspace_pair[..., 0], kspace_pair[..., 1]
    kspace_quotient = safe_divide(kspace_out, kspace_in, thresh=1e-3)  # TODO set thresh adaptively, say based on noise std?
    psf = np.real(sp.ifft(kspace_quotient))
    return psf

def forward_model(input_kspace, psf_shape):
    shape = input_kspace.shape
    Z = sp.linop.Resize(shape, psf_shape)
    F = sp.linop.FFT(shape)
    D = sp.linop.Multiply(shape, input_kspace)
    return D * F * Z

def estimate_psf_reg(patch_pair, psf_shape=reg_psf_shape, lamda=1e-2, tol=1e-8, max_iter=1e5, verbose=False):
    kspace_pair = sp.fft(patch_pair, axes=(0, 1, 2))
    kspace_in, kspace_out = kspace_pair[..., 0], kspace_pair[..., 1]
    A = forward_model(kspace_in, psf_shape)
    y = kspace_out
    app = sp.app.LinearLeastSquares(A, y, x=np.zeros(A.ishape, dtype=np.complex128), tol=tol, max_iter=max_iter, show_pbar=verbose, lamda=lamda)
    soln = app.run()
    psf = np.abs(soln)  # was real before, does that make more sense?
    return psf

def get_FWHM_from_image(psf, num_workers, stride=1, batch_axis=2):
    func = get_FWHM_from_pixel
    patch_shape = (1, 1, 1)
    out_shape = (3,)
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
    i_half_max_1 = find_root(i_half_max,
                             x[i_half_max] - half_max,
                             i_half_max + 1,
                             x[i_half_max + 1] - half_max)
    i_half_max = i_max + np.argmin(x[i_max::1] > half_max)  # just right of half-max
    i_half_max_2 = find_root(i_half_max - 1,
                             x[i_half_max - 1] - half_max,
                             i_half_max,
                             x[i_half_max] - half_max)
    return i_half_max_2 - i_half_max_1

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

def get_resolution_mask(reference, target, metal=False):
    mask_empty = analysis.get_mask_empty(reference)
    mask_implant = analysis.get_mask_implant(mask_empty)
    if metal:
        mask_signal = analysis.get_mask_signal(reference)
        signal_ref = analysis.get_typical_level(reference, mask_signal, mask_implant)
        error = target - reference 
        normalized_error = safe_divide(error, signal_ref)
        mask_artifact = analysis.get_mask_artifact(normalized_error)
        mask = analysis.get_mask_register(mask_empty, mask_implant, mask_artifact)
    else:
        mask_lattice = analysis.get_mask_lattice(reference)
        mask = np.logical_and(mask_lattice, ~mask_implant)
    return mask

def map_resolution(reference, target, unit_cell, stride=8, num_workers=8, mask=None):
    num_dims = target.ndim
    patch_shape = (unit_cell,) * num_dims # might want to double for better noise robustness
    if mask is None:
        mask = get_resolution_mask(reference, target)
    psf = estimate_psf(reference, target, mask, patch_shape, stride, num_workers)
    fwhm = get_FWHM_from_image(psf, num_workers)
    return psf, fwhm
