from multiprocessing import Pool
import numpy as np
from scipy.ndimage import map_coordinates

def interpolate(psf, factor):
    nx, ny, nz = psf.shape
    coords = np.mgrid[:nx:1/factor, :ny:1/factor, :nz:1/factor]
    return map_coordinates(psf, coords, order=2)

def get_FWHM_in_parallel(psf, num_workers=8):
    psf_splits = np.array_split(psf, num_workers, axis=0)
    with Pool(num_workers) as p:
        result = p.map(get_FWHM_from_many_psf_3D, psf_splits) 
        # result = p.map(get_FWHM_from_many_psf_2D, psf_splits) 
    result = np.concatenate(result, axis=0)
    return result

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
    if psf[max_idx] == 0:
        return 0, 0, 0
    fwhm_x = fwhm(psf[:, max_idx[1], max_idx[2]], im=max_idx[0])
    fwhm_y = fwhm(psf[max_idx[0], :, max_idx[2]], im=max_idx[1])
    fwhm_z = fwhm(psf[max_idx[0], max_idx[1], :], im=max_idx[2])
    return fwhm_x, fwhm_y, fwhm_z

def get_FWHM_from_psf_2D(psf):
    psf = np.abs(psf)
    max_idx = np.unravel_index(np.argmax(psf), psf.shape)
    if psf[max_idx] == 0:
        return 0, 0
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
    if y1 == y2:
        print('find_root division by zero')
    return x1 - y1 * (x2 - x1) / (y2 - y1)
