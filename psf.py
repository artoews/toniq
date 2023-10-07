import numpy as np
from scipy.ndimage import map_coordinates
import sigpy as sp
from multiprocessing import Pool


def model(image, psf_size, ndim):
    # zero-pad PSF to match image, allowing multiplication in k-space
    if ndim == 3:
        Z = sp.linop.Resize(image.shape, (psf_size,) * ndim)  
    elif ndim == 2:
        Z = sp.linop.Resize(image.shape, (psf_size, psf_size, 1)) * sp.linop.Reshape((psf_size, psf_size, 1), (psf_size, psf_size))
    F = sp.linop.FFT(image.shape)
    D = sp.linop.Multiply(image.shape, sp.fft(image))
    C = sp.linop.Resize(shape_without_margin(image.shape, psf_size), image.shape) # crop away margin to remove wrapped info from circular convolution
    return C * F.H * D * F * Z

def shape_without_margin(shape, margin):
    return tuple(n - margin for n in shape)

def estimate_psf(image_ref, image_blur, patch_size, psf_size, ndim, psf_init=None, start=None):
    ndim = image_ref.ndim
    patch = tuple(slice(start[i], start[i] + patch_size + psf_size) for i in range(ndim))
    image_ref_patch = image_ref[patch]
    image_blur_patch = image_blur[patch]
    psf_soln = estimate_psf_patch(image_ref_patch, image_blur_patch, psf_size, ndim, psf_init=psf_init)
    return image_ref_patch, image_blur_patch, psf_init, psf_soln

def estimate_psf_all_3d(image_ref, image_blurred, patch_size, psf_size, stride, tol=1e-6, max_iter=1000):
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
                soln = estimate_psf_patch(image_ref[patch], image_blurred[patch], psf_size, tol, 3, psf_init=zero_init, max_iter=max_iter)
                psf[ix // stride, iy // stride, iz // stride] = soln
    return psf

def estimate_psf_all_2d(image_ref, image_blurred, patch_size, psf_size, stride, tol=1e-6, max_iter=1000):
    nx, ny, nz = image_ref.shape
    zero_init = np.zeros((psf_size, psf_size), dtype=np.complex128)
    nx_pts = np.arange(0, nx - patch_size - psf_size, stride)
    ny_pts = np.arange(0, ny - patch_size - psf_size, stride)
    nz_pts = np.arange(0, nz - patch_size - 1, stride)
    psf = np.zeros((len(nx_pts), len(ny_pts), len(nz_pts)) + zero_init.shape, dtype=np.complex128)
    for ix in nx_pts:
        for iy in ny_pts:
            for iz in nz_pts:
                patch = (slice(ix, ix + patch_size + psf_size), slice(iy, iy + patch_size + psf_size), slice(iz, iz + patch_size + 1))
                soln = estimate_psf_patch(image_ref[patch], image_blurred[patch], psf_size, tol, 2, psf_init=zero_init, max_iter=max_iter)
                psf[ix // stride, iy // stride, iz // stride] = soln
    return psf


def split_volumes_with_overlap(volumes, num_workers, overlap, stride, axis):
    size = volumes[0].shape[axis]
    pts = np.arange(0, size - overlap, stride)
    pts = np.array_split(pts, num_workers)
    splits = []
    for i in np.arange(num_workers):
        start = pts[i][0]
        end = pts[i][-1] + overlap
        splits.append(np.arange(start, end + 1))
    split_volumes = []
    for vol in volumes:
        split_vol = tuple(np.take(vol, split, axis=axis) for split in splits)
        split_volumes.append(split_vol)
    return split_volumes

def estimate_psf_all_in_parallel(image_ref, image_blurred, patch_size, stride, psf_size=5, num_workers=8, split_axis=2):
    sub_images_ref, sub_images_blurred = split_volumes_with_overlap((image_ref, image_blurred), num_workers, patch_size + psf_size, stride, split_axis)
    inputs = list(zip(
                sub_images_ref,
                sub_images_blurred,
                (patch_size,) * num_workers,
                (psf_size,) * num_workers,
                (stride,) * num_workers
                ))
    with Pool(num_workers) as p:
        result = p.starmap(estimate_psf_all_2d, inputs)
    result = np.concatenate(result, axis=split_axis)
    return result

def get_FWHM_in_parallel(psf, num_workers=8):
    psf_splits = np.array_split(psf, num_workers, axis=0)
    with Pool(num_workers) as p:
        # result = p.map(get_FWHM_from_many_psf_3D, psf_splits) 
        result = p.map(get_FWHM_from_many_psf_2D, psf_splits) 
    result = np.concatenate(result, axis=0)
    return result

def estimate_psf_patch(image_ref, image_blurred, psf_size, tol, ndim, psf_init=None, max_iter=1000, verbose=False):
    if psf_init is None:
        psf_init = np.zeros((psf_size,) * ndim, dtype=np.complex128)
    op = model(image_ref, psf_size, ndim)
    patch_shape = shape_without_margin(image_ref.shape, psf_size)
    image_ref = sp.util.resize(image_ref, patch_shape)
    image_blurred = sp.util.resize(image_blurred, patch_shape)
    app = sp.app.LinearLeastSquares(op, image_blurred, x=psf_init.copy(), tol=tol, max_iter=max_iter, show_pbar=verbose)
    return app.run()

def interpolate(psf, factor):
    nx, ny, nz = psf.shape
    coords = np.mgrid[:nx:1/factor, :ny:1/factor, :nz:1/factor]
    return map_coordinates(psf, coords, order=2)

def get_FWHM_from_many_psf_3D(psf):
    nx, ny, nz = psf.shape[:3]
    fwhm = np.zeros((nx, ny, nz, 3))
    for ix in np.arange(nx): 
        for iy in np.arange(ny):
            for iz in np.arange(nz):
                fwhm[ix, iy, iz, :] = get_FWHM_from_psf_3D(psf[ix, iy, iz, ...])
    return fwhm

def get_FWHM_from_many_psf_2D(psf):
    nx, ny, nz = psf.shape[:3]
    fwhm = np.zeros((nx, ny, nz, 2))
    for ix in np.arange(nx): 
        for iy in np.arange(ny):
            for iz in np.arange(nz):
                fwhm[ix, iy, iz, :] = get_FWHM_from_psf_2D(psf[ix, iy, iz, ...])
    return fwhm

def get_FWHM_from_psf_3D(psf):
    psf = np.abs(psf)
    max_idx = np.unravel_index(np.argmax(psf), psf.shape)
    fwhm_x = fwhm(psf[:, max_idx[1], max_idx[2]], im=max_idx[0])
    fwhm_y = fwhm(psf[max_idx[0], :, max_idx[2]], im=max_idx[1])
    fwhm_z = fwhm(psf[max_idx[0], max_idx[1], :], im=max_idx[2])
    return fwhm_x, fwhm_y, fwhm_z

def get_FWHM_from_psf_2D(psf):
    psf = np.abs(psf)
    max_idx = np.unravel_index(np.argmax(psf), psf.shape)
    fwhm_x = fwhm(psf[:, max_idx[1]], im=max_idx[0])
    fwhm_y = fwhm(psf[max_idx[0], :], im=max_idx[1])
    return fwhm_x, fwhm_y

def fwhm(x, im=None):
    if im is None:
        im = np.argmax(x)
    hm = x[im] / 2
    if im < 1 or im >= len(x) - 1:
        return 0
    i_hm = im - np.argmin(x[im::-1] > hm)  # just left of half-max
    i_hm_1 = find_root(i_hm, x[i_hm] - hm, i_hm + 1, x[i_hm + 1] - hm)
    i_hm = im + np.argmin(x[im::1] > hm)  # just right of half-max
    i_hm_2 = find_root(i_hm - 1, x[i_hm - 1] - hm, i_hm, x[i_hm] - hm)
    return i_hm_2 - i_hm_1

def find_root(x1, y1, x2, y2):
    # find zero-crossing of line through points (x1, y1) and (x2, y2)
    return x1 - y1 * (x2 - x1) / (y2 - y1)
