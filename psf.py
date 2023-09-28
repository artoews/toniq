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
    psf = np.zeros((len(nx_pts), len(ny_pts), len(nz_pts)) + zero_init.shape)
    for ix in nx_pts:
        for iy in ny_pts:
            for iz in nz_pts:
                print(ix, iy, iz)
                patch = tuple(slice(i, i + patch_size + psf_size) for i in (ix, iy, iz))
                soln = estimate_psf_patch(image_ref[patch], image_blurred[patch], psf_size, psf_init=zero_init)
                psf[ix // stride, iy // stride, iz // stride] = soln
    return psf

# def estimimate_psf_all_in_parallel(image_ref, image_blurred, patch_size, psf_size, stride, accel=8, tol=1e-2, max_iter=100):
#     sub_images_ref = np.split(image_ref, accel, axis=0)
#     # setup pool
#     # collect results
# 
# # TODO in another function, compute the FWHM and collect into an array. Interpolate to get a map at the image resolution.

def estimate_psf_patch(image_ref, image_blurred, psf_size, psf_init=None, tol=1e-2, max_iter=100):
    ndim =image_ref.ndim
    if psf_init is None:
        psf_init = np.zeros((psf_size,) * ndim, dtype=np.complex128)
    op = model(image_ref, psf_size)
    patch_shape = shape_without_margin(image_ref.shape, psf_size)
    image_ref = sp.util.resize(image_ref, patch_shape)
    image_blurred = sp.util.resize(image_blurred, patch_shape)
    app = sp.app.LinearLeastSquares(op, image_blurred, x=psf_init.copy(), tol=tol, max_iter=max_iter)
    return app.run()

def interpolate_sinc(psf, size):
    shape = (size,) * psf.ndim
    factor = np.prod(shape) / np.prod(psf.shape)
    return sp.ifft(sp.resize(sp.fft(psf), shape)) * np.sqrt(factor)

def get_FWHM(x):
    peak_idx = np.argmax(x)
    half_max = x[peak_idx] / 2
    # print('FWHM')
    # print(x[peak_idx::1])
    # print(x[peak_idx::-1])
    return np.argmin(x[peak_idx::1] > half_max) + np.argmin(x[peak_idx::-1] > half_max)

# TODO interpolate along an arbitrary direction of the PSF to measure FWHM in that direction
