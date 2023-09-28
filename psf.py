import numpy as np
import sigpy as sp


def model(image, psf_size):
    Z = sp.linop.Resize(image.shape, (psf_size,) * image.ndim)
    F = sp.linop.FFT(image.shape)
    D = sp.linop.Multiply(image.shape, sp.fft(image))
    C = sp.linop.Resize(shape_without_margin(image.shape, psf_size), image.shape)
    return C * F.H * D * F * Z

def shape_without_margin(shape, margin):
    return tuple(n - margin for n in shape)

def estimate_psf(image_ref, image_blur, patch_size, psf_size, psf_init=None, start=(50, 70, 31)):
    ndim = image_ref.ndim
    patch = tuple(slice(start[i], start[i] + patch_size + psf_size) for i in range(ndim))
    if psf_init is None:
        psf_init = np.zeros((psf_size,) * ndim, dtype=np.complex128)
    # init[patch_size//2, patch_size//2, patch_size//2] = 1
    image_ref_patch = image_ref[patch]
    image_blur_patch = image_blur[patch]
    op = model(image_ref_patch, psf_size)
    image_ref_patch = sp.util.resize(image_ref_patch, shape_without_margin(image_ref_patch.shape, psf_size))
    image_blur_patch = sp.util.resize(image_blur_patch, shape_without_margin(image_blur_patch.shape, psf_size))
    app = sp.app.LinearLeastSquares(op, image_blur_patch, x=psf_init.copy(), tol=1e-2, max_iter=1e2)
    psf_soln =app.run()
    return image_ref_patch, image_blur_patch, psf_init, psf_soln

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
