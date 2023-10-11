import itertools
import numpy as np
import sigpy as sp
from multiprocessing import Pool

def batch_with_overlap(volume, overlap, num_batches, axis=-1):
    starts = np.arange(0, volume.shape[axis] - overlap)
    starts_of_batches = np.array_split(starts, num_batches)
    indices_of_batches = tuple(np.arange(starts_of_batch[0], starts_of_batch[-1] + overlap + 1) for starts_of_batch in starts_of_batches)
    return tuple(np.take(volume, indices_of_batch, axis=axis) for indices_of_batch in indices_of_batches)

def map_psf(image_in, image_out, mask, patch_shape, psf_shape, stride, mode, num_workers=1):
    # batched dimension is not strided
    batch_axis = 2

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
    psf = np.zeros(num_locs + psf_shape, dtype=np.complex128)
    for patch_loc in itertools.product(*patch_locs):
        slc = tuple(slice(patch_loc[i], patch_loc[i] + patch_shape[i] + psf_shape[i] - 1) for i in range(3))
        patch_in = image_in[slc]
        patch_out = image_out[slc]
        if mask is not None and not np.all(mask[slc]):
            soln = psf_init
        elif mode == 'iterative':
            tol = 1e-6
            max_iter = 1e3
            soln = estimate_psf_iterative(patch_in, patch_out, psf_shape, psf_init, tol, max_iter)
        elif mode == 'direct':
            pass
            # soln = estimate_psf_direct(patch_in, patch_out, psf_shape)
        idx = tuple(patch_loc[i] // strides[i] for i in range(3))
        psf[idx] = soln
    return psf

def estimate_psf_iterative(image_in, image_out, psf_shape, psf_init, tol, max_iter, verbose=False):
    # image_in and image_out having a margin to accomodate circular convolution
    A = model(image_in, psf_shape)
    y = A.linops[0](image_out)  # crop away margin to match format of model output
    app = sp.app.LinearLeastSquares(A, y, x=psf_init.copy(), tol=tol, max_iter=max_iter, show_pbar=verbose)
    return app.run()

def model(image, psf_shape):
    # zero-pad PSF to match image, allowing multiplication in k-space
    Z = sp.linop.Resize(image.shape, psf_shape)  
    F = sp.linop.FFT(image.shape)
    D = sp.linop.Multiply(image.shape, sp.fft(image))
    C = sp.linop.Resize(shape_without_margin(image.shape, psf_shape), image.shape) # crop away margin to remove wrapped info from circular convolution
    return C * F.H * D * F * Z

def shape_without_margin(shape, margin):
    return tuple(s - m for s, m in zip(shape, margin))
