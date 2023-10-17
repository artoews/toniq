import itertools
import numpy as np
from scipy.linalg import circulant, lstsq
from scipy.signal import convolve2d
import sigpy as sp
from multiprocessing import Pool
from util import safe_divide

def batch_with_overlap(volume, overlap, num_batches, axis=-1):
    starts = np.arange(0, volume.shape[axis] - overlap)
    starts_of_batches = np.array_split(starts, num_batches)
    indices_of_batches = tuple(np.arange(starts_of_batch[0], starts_of_batch[-1] + overlap + 1) for starts_of_batch in starts_of_batches)
    return tuple(np.take(volume, indices_of_batch, axis=axis) for indices_of_batch in indices_of_batches)

def map_psf(image_in, image_out, mask, patch_shape, psf_shape, stride, mode, num_workers=1):
    # batched dimension is not strided
    batch_axis = 2
    if mode == 'kspace':
        psf_shape = (1, 1, 1)

    if image_in.shape != image_out.shape or mask.shape != image_out.shape:
        raise ValueError('Image input, output and mask must have same shape; got {}, {}, {}'.format(image_in.shape, image_out.shape, mask.shape))

    if num_workers > 1:
        batched_image_in = batch_with_overlap(image_in, patch_shape[2] + psf_shape[2] - 1, num_workers, axis=batch_axis)
        batched_image_out = batch_with_overlap(image_out, patch_shape[2] + psf_shape[2] - 1, num_workers, axis=batch_axis)
        batched_mask = batch_with_overlap(mask, patch_shape[2] + psf_shape[2] - 1, num_workers, axis=batch_axis)
        inputs = list(zip(
                    batched_image_in,
                    batched_image_out,
                    batched_mask,
                    (patch_shape,) * num_workers,
                    (psf_shape,) * num_workers,
                    (stride,) * num_workers,
                    (mode,) * num_workers
                    ))
        with Pool(num_workers) as p:
            result = p.starmap(map_psf, inputs)
        psf = np.concatenate(result, axis=batch_axis)
        return psf

    if mode == 'iterative':
        psf_init = np.zeros(psf_shape, dtype=np.complex128)
    strides = np.roll((1, stride, stride), batch_axis)
    patch_locs = tuple(np.arange(0, image_in.shape[i] - patch_shape[i] - psf_shape[i] + 1, strides[i]) for i in range(3))
    num_locs = tuple(len(p) for p in patch_locs)
    if mode == 'kspace':
        psf_shape = patch_shape
    psf = np.zeros(num_locs + psf_shape, dtype=np.complex128)
    if mode == 'kspace':
        psf_shape = (1, 1, 1)
    for patch_loc in itertools.product(*patch_locs):
        slc = tuple(slice(patch_loc[i], patch_loc[i] + patch_shape[i] + psf_shape[i] - 1) for i in range(3))
        patch_in = image_in[slc]
        patch_out = image_out[slc]
        if mask is not None and not np.all(mask[slc]):
            soln = np.zeros(psf_shape, dtype=np.complex128)
        elif mode == 'iterative':
            tol = 1e-4  # was 1e-6
            max_iter = 1e3
            soln = estimate_psf_iterative(patch_in, patch_out, psf_shape, psf_init, tol, max_iter)
        elif mode == 'direct':
            soln = estimate_psf_direct(patch_in, patch_out, psf_shape)
        elif mode == 'kspace':
            soln = estimate_psf_kspace(patch_in, patch_out)
        idx = tuple(patch_loc[i] // strides[i] for i in range(3))
        psf[idx] = soln
    return psf

def estimate_psf_kspace(patch_in, patch_out):
    # TODO set thresh based on noise std?
    kspace_in = sp.fft(patch_in)
    kspace_out = sp.fft(patch_out)
    kspace_quotient = safe_divide(kspace_out, kspace_in, thresh=1e-2)
    psf = np.abs(sp.ifft(kspace_quotient))
    return psf

def estimate_psf_iterative(image_in, image_out, psf_shape, psf_init, tol, max_iter, verbose=True):
    # image_in and image_out having a margin to accomodate circular convolution
    A = model_op(image_in, psf_shape)
    # A = model_matrix_op(image_in, psf_shape)  # slower
    y = sp.resize(image_out, shape_without_margin(image_in.shape, psf_shape))  # crop away margin to match format of model output
    app = sp.app.LinearLeastSquares(A, y, x=psf_init.copy(), tol=tol, max_iter=max_iter, show_pbar=verbose)
    return app.run()

def estimate_psf_direct(image_in, image_out, psf_shape):
    A, _, _ = model_matrix(image_in, psf_shape)
    y = sp.resize(image_out, shape_without_margin(image_in.shape, psf_shape))  # crop away margin to match format of model output
    p, res, rnk, s = lstsq(A, y.ravel())
    return p.reshape(psf_shape)

def model_op(image, psf_shape):
    # zero-pad PSF to match image, allowing multiplication in k-space
    Z = sp.linop.Resize(image.shape, psf_shape)  
    F = sp.linop.FFT(image.shape)
    D = sp.linop.Multiply(image.shape, sp.fft(image))
    C = sp.linop.Resize(shape_without_margin(image.shape, psf_shape), image.shape) # crop away margin to remove wrapped info from circular convolution
    return C * F.H * D * F * Z

def shape_without_margin(shape, margin):
    # TODO rename so the + 1 in here isn't a surprise - this is specifically for removing the invalid entries after circular convolution
    return tuple(s - m + 1 for s, m in zip(shape, margin))

def model_matrix(image, psf_shape):
    pad_shape = tuple(n + m - 1 for n, m in zip(image.shape, psf_shape))
    out_shape = tuple(n - m + 1 for n, m in zip(image.shape, psf_shape))
    image_padded = sp.resize(image, pad_shape)
    image_padded = back_pad_to(image, pad_shape)
    convolution_matrix = circulant(image_padded.ravel())
    input_pad_mask = back_pad_to(np.ones(psf_shape), pad_shape).ravel().astype(bool)
    output_crop_mask = back_pad_to(np.ones(out_shape), pad_shape).ravel().astype(bool)
    output_crop_mask = np.roll(output_crop_mask, np.ravel_multi_index(tuple(n-1 for n in psf_shape), pad_shape))
    convolution_matrix = convolution_matrix[:, input_pad_mask]
    convolution_matrix = convolution_matrix[output_crop_mask, :]
    return convolution_matrix, out_shape, pad_shape

def model_matrix_op(image, psf_shape):
    matrix, out_shape, _ = model_matrix(image, psf_shape)
    vec_shape = (np.prod(psf_shape), 1)
    R_in = sp.linop.Reshape(vec_shape, psf_shape)
    M = sp.linop.MatMul(vec_shape, matrix)
    R_out = sp.linop.Reshape(out_shape, M.oshape)
    return R_out * M * R_in

def back_pad_to(arr, new_shape):
    return np.pad(arr, tuple((0, n - m) for n, m in zip(new_shape, arr.shape)))

def check_model_matrix(image, psf):
    matrix, out_shape, pad_shape = model_matrix(image, psf.shape)
    # input = sp.resize(psf, pad_shape).ravel()
    # matrix_result = matrix.dot(input)
    matrix_result = matrix.dot(psf.ravel()).reshape(out_shape)
    correct_result = convolve2d(image, psf, mode='valid')
    # correct_result = convolve2d(image, psf, mode='full', boundary='wrap')
    print('image \n', image)
    print('psf\n', psf)
    print('compressed matrix\n', matrix)
    print('matrix result\n', matrix_result)
    print('correct result\n', correct_result)
    print('error', np.sum(np.abs(matrix_result - correct_result)))
    assert np.array_equal(matrix_result, correct_result)
    # TODO check it works in 3D with scipy.convolve