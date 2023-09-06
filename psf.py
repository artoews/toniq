import numpy as np
import sigpy as sp


def model(image, psf_size):
    Z = pad_psf_op(image.shape, psf_size)
    F = sp.linop.FFT(image.shape)
    D = sp.linop.Multiply(image.shape, sp.fft(image))
    C = crop_wrap_op(image.shape, psf_size)
    return C * F.H * D * F * Z

def crop_wrap_op(ishape, psf_size):
    idx = tuple(slice(psf_size // 2, n - psf_size // 2) for n in ishape)
    return sp.linop.Slice(ishape, idx)

def pad_psf_op(shape, psf_size):
    idx = tuple(slice(n // 2 - psf_size // 2, n // 2 + psf_size // 2) for n in shape)
    return sp.linop.Embed(shape, idx)

def estimate_psf(image_ref, image_blur, patch_size, psf_size, l2=1e-5):
    start = (50, 70, 30)
    ndim = image_ref.ndim
    patch = tuple(slice(start[i], start[i] + patch_size + psf_size) for i in range(ndim))
    psf_init = np.zeros((psf_size,) * ndim, dtype=np.complex128)
    # init[patch_size//2, patch_size//2, patch_size//2] = 1
    image_ref_patch = image_ref[patch]
    image_blur_patch = image_blur[patch]
    C = crop_wrap_op(image_blur_patch.shape, psf_size)
    op = model(image_ref_patch, psf_size)
    app = sp.app.LinearLeastSquares(op, C(image_blur_patch), x=psf_init.copy(), lamda=l2)
    psf_soln =app.run()
    return image_ref_patch, image_blur_patch, psf_init, psf_soln

