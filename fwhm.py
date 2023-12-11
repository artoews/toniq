from multiprocessing import Pool
import numpy as np
import sigpy as sp
from psf import generic_filter

def get_FWHM_from_image_new(psf, num_workers, stride=1, batch_axis=2):
    # TODO figure out why this isn't working
    func = get_FWHM_from_pixel
    patch_shape = (1, 1, 1)
    out_shape = (3,)
    return generic_filter(psf, func, patch_shape, out_shape, stride, batch_axis, num_batches=num_workers)

def get_FWHM_from_image(psf, num_workers=1):
    if num_workers > 1:
        psf_splits = np.array_split(psf, num_workers, axis=0)
        with Pool(num_workers) as p:
            result = p.map(get_FWHM_from_image, psf_splits) 
        result = np.concatenate(result, axis=0)
    else:
        nx, ny, nz = psf.shape[:3]
        ndim = len(psf.shape[3:])
        fwhm = np.zeros((nx, ny, nz, ndim))
        for ix in np.arange(nx): 
            for iy in np.arange(ny):
                for iz in np.arange(nz):
                    fwhm[ix, iy, iz, :] = get_FWHM_from_pixel(psf[ix, iy, iz, ...])
        result = fwhm
    return result

def get_FWHM_from_pixel(psf):
    psf = np.abs(psf)
    ndim = np.squeeze(psf).ndim
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

def sinc(in_shape, out_shape):
    k = sp.resize(np.ones(out_shape), in_shape)
    psf = sp.ifft(k)
    return get_FWHM_from_pixel(psf)
