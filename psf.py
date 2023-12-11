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

def model(input_kspace, psf_shape):
    shape = input_kspace.shape
    Z = sp.linop.Resize(shape, psf_shape)
    F = sp.linop.FFT(shape)
    D = sp.linop.Multiply(shape, input_kspace)
    return D * F * Z

def estimate_psf_reg(patch_pair, psf_shape=reg_psf_shape, lamda=1e-2, tol=1e-8, max_iter=1e5, verbose=False):
    kspace_pair = sp.fft(patch_pair, axes=(0, 1, 2))
    kspace_in, kspace_out = kspace_pair[..., 0], kspace_pair[..., 1]
    A = model(kspace_in, psf_shape)
    y = kspace_out
    app = sp.app.LinearLeastSquares(A, y, x=np.zeros(A.ishape, dtype=np.complex128), tol=tol, max_iter=max_iter, show_pbar=verbose, lamda=lamda)
    soln = app.run()
    psf = np.abs(soln)  # was real before, does that make more sense?
    return psf
