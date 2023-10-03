import numpy as np
import sigpy as sp
from multiprocessing import Pool


def model(image, psf_size):
    Z = sp.linop.Resize(image.shape, (psf_size,) * image.ndim)
    F = sp.linop.FFT(image.shape)
    D = sp.linop.Multiply(image.shape, sp.fft(image))
    C = sp.linop.Resize(shape_without_margin(image.shape, psf_size), image.shape)
    return C * F.H * D * F * Z

def shape_without_margin(shape, margin):
    return tuple(n - margin for n in shape)

def estimate_psf(image_ref, image_blur, patch_size, psf_size, psf_init=None, start=None):
    ndim = image_ref.ndim
    patch = tuple(slice(start[i], start[i] + patch_size + psf_size) for i in range(ndim))
    image_ref_patch = image_ref[patch]
    image_blur_patch = image_blur[patch]
    psf_soln = estimate_psf_patch(image_ref_patch, image_blur_patch, psf_size, psf_init=psf_init)
    return image_ref_patch, image_blur_patch, psf_init, psf_soln

def estimate_psf_all(image_ref, image_blurred, patch_size, psf_size, stride, tol=1e-2, max_iter=100):
    nx, ny, nz = image_ref.shape
    zero_init = np.zeros((psf_size,) * 3, dtype=np.complex128)
    nx_pts = np.arange(0, nx - patch_size - psf_size, stride)
    ny_pts = np.arange(0, ny - patch_size - psf_size, stride)
    nz_pts = np.arange(0, nz - patch_size - psf_size, stride)
    psf = np.zeros((len(nx_pts), len(ny_pts), len(nz_pts)) + zero_init.shape, dtype=np.complex128)
    for ix in nx_pts:
        for iy in ny_pts:
            for iz in nz_pts:
                patch = tuple(slice(i, i + patch_size + psf_size) for i in (ix, iy, iz))
                soln = estimate_psf_patch(image_ref[patch], image_blurred[patch], psf_size, psf_init=zero_init, tol=tol, max_iter=max_iter)
                psf[ix // stride, iy // stride, iz // stride] = soln
    return psf

def estimate_psf_all_in_parallel(image_ref, image_blurred, patch_size, psf_size, stride, num_workers=8):
    # TODO in another function (?), compute the FWHM and collect into an array. Interpolate to get a map at the image resolution.
    nx = image_ref.shape[0]
    nx_pts = np.arange(0, nx - patch_size - psf_size, stride)
    pt_splits = np.array_split(nx_pts, num_workers)
    splits = []
    for i in np.arange(num_workers):
        start = pt_splits[i][0]
        end = pt_splits[i][-1] + patch_size + psf_size
        splits.append(np.arange(start, end + 1))
    sub_images_ref = (np.take(image_ref, split, axis=0) for split in splits)
    sub_images_blurred = (np.take(image_blurred, split, axis=0) for split in splits)
    inputs = list(zip(
                sub_images_ref,
                sub_images_blurred,
                (patch_size,) * num_workers,
                (psf_size,) * num_workers,
                (stride,) * num_workers
                ))
    with Pool(num_workers) as p:
        result = p.starmap(estimate_psf_all, inputs)
    result = np.concatenate(result, axis=0)
    return result

def get_FWHM_in_parallel(psf, num_workers=8):
    psf_splits = np.array_split(psf, num_workers, axis=0)
    with Pool(num_workers) as p:
        result = p.map(get_FWHM_from_many_psf, psf_splits) 
    result = np.concatenate(result, axis=0)
    return result

def estimate_psf_patch(image_ref, image_blurred, psf_size, psf_init=None, tol=1e-2, max_iter=100, verbose=False):
    ndim =image_ref.ndim
    if psf_init is None:
        psf_init = np.zeros((psf_size,) * ndim, dtype=np.complex128)
    op = model(image_ref, psf_size)
    patch_shape = shape_without_margin(image_ref.shape, psf_size)
    image_ref = sp.util.resize(image_ref, patch_shape)
    image_blurred = sp.util.resize(image_blurred, patch_shape)
    app = sp.app.LinearLeastSquares(op, image_blurred, x=psf_init.copy(), tol=tol, max_iter=max_iter, show_pbar=verbose)
    return app.run()

def interpolate_sinc(psf, size):
    shape = (size,) * psf.ndim
    factor = np.prod(shape) / np.prod(psf.shape)
    return sp.ifft(sp.resize(sp.fft(psf), shape)) * np.sqrt(factor)

def get_FWHM_from_many_psf(psf):
    nx, ny, nz = psf.shape[:3]
    fwhm = np.zeros((nx, ny, nz, 3))
    for ix in np.arange(nx): 
        for iy in np.arange(ny):
            for iz in np.arange(nz):
                fwhm[ix, iy, iz, :] = get_FWHM_from_psf(psf[ix, iy, iz, ...])
    return fwhm

def get_FWHM_from_psf(psf, interp_factor=8):
    psf = np.abs(psf)
    psf_size = psf.shape[0]
    psf_int = interpolate_sinc(psf, psf_size * interp_factor)  # TODO consider another interpolation strategy, like use sigpy.interpolate with b splines
    max_idx = np.unravel_index(np.argmax(psf_int), psf_int.shape)
    max_val = psf_int[max_idx]
    fwhm_x = fwhm(psf_int[:, max_idx[1], max_idx[2]], max_val) / interp_factor
    fwhm_y = fwhm(psf_int[max_idx[0], :, max_idx[2]], max_val) / interp_factor
    fwhm_z = fwhm(psf_int[max_idx[0], max_idx[1], :], max_val) / interp_factor
    return fwhm_x, fwhm_y, fwhm_z

def fwhm(x, max_val=None, max_idx=None):
    if max_idx is None or max_val is None:
        max_idx = np.argmax(x)
        max_val = x[max_idx]
    half_max = max_val / 2
    return np.argmin(x[max_idx::1] > half_max) + np.argmin(x[max_idx::-1] > half_max)

def get_FWHM(x):
    peak_idx = np.argmax(x)
    half_max = x[peak_idx] / 2
    # print('FWHM')
    # print(x[peak_idx::1])
    # print(x[peak_idx::-1])
    return np.argmin(x[peak_idx::1] > half_max) + np.argmin(x[peak_idx::-1] > half_max)
