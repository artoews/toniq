"""Functions for spatial resolution (SR) mapping & plotting.

"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy.typing as npt
import numpy as np
import sigpy as sp
import functools
from scipy.signal import unit_impulse
import scipy.ndimage as ndi

from toniq.filter import generic_filter
from toniq.linop import get_matrix
from toniq.plot import overlay_mask, colorbar_axis
from toniq.plot_params import *


def get_map(
        reference: npt.NDArray[np.float64],
        target: npt.NDArray[np.float64], 
        psf_shape: tuple[int], 
        patch_shape: tuple[int], 
        resolution_mm: tuple[float], 
        mask: npt.NDArray[np.bool], 
        stride: int, 
        num_workers: int = 1
        ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Get SR map.

    The SR map is computed by measuring the degree of blurring in a target image with respect to a reference image.
    Blurring is modelled as a locally linear operation whereby corresponding 3D patches of the target and reference images
    are related by convolution with an unknown point spread function (PSF).
    The SR map results from measuring the FWHM of the local PSF for each patch.

    Args:
        reference (npt.NDArray[np.float64]): reference image
        target (npt.NDArray[np.float64]): target image
        psf_shape (tuple[int]): PSF footprint
        patch_shape (tuple[int]): patch size used for local PSF estimation
        resolution_mm (tuple[float]): resolution in x, y, z
        mask (npt.NDArray[np.bool]): mask restricting the domain for resolution analysis
        stride (int): stride between patches
        num_workers (int, optional): parallelization factor. Defaults to 1.

    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: [map of PSF, map of FWHM in mm]
    """
    psf = estimate_psf(reference, target, mask, psf_shape, patch_shape, stride, num_workers)
    fwhm = get_FWHM_from_image(psf, psf_shape, num_workers)
    for i in range(fwhm.shape[-1]):
        fwhm[..., i] = fwhm[..., i] * resolution_mm[i]
    if stride == 1:
        psf = sp.resize(psf, target.shape[:3] + psf.shape[3:])
        fwhm = sp.resize(fwhm, target.shape[:3] + fwhm.shape[3:])
    return psf, fwhm

def estimate_psf(
        image_in: npt.NDArray[np.float64],
        image_out: npt.NDArray[np.float64],
        mask: npt.NDArray[np.bool],
        psf_shape: tuple[int],
        patch_shape: tuple[int],
        stride: int,
        num_batches: int
        ) -> npt.NDArray[np.float64]:
    """Local PSF estimation for an image pair.

    For each patch, a local PSF is estimated by means of least squares estimation.

    Args:
        image_in (npt.NDArray[np.float64]): input image
        image_out (npt.NDArray[np.float64]): output image
        mask (npt.NDArray[np.bool]): mask excluding regions lacking structure or having excessive artifact
        psf_shape (tuple[int]): PSF footprint
        patch_shape (tuple[int]): patch size used for local PSF estimation
        stride (int): stride between patches
        num_batches (int): parallelization factor

    Returns:
        npt.NDArray[np.float64]: map of the local PSF for each patch
    """
    images_stack = np.stack((image_in, image_out), axis=-1)
    images_stack[~mask, ...] = np.nan
    batch_axis = 2
    func = functools.partial(deconvolve_by_model, psf_shape)
    return generic_filter(images_stack, func, patch_shape, psf_shape, stride, batch_axis, num_batches=num_batches)

def deconvolve_by_model(
        psf_shape: tuple[int],
        patch_pair: npt.NDArray[np.float64],
        verbose: bool = False
        ) -> npt.NDArray[np.float64]:
    """Local PSF estimation for one patch.

    The inverse problem is solved iteratively by means of Conjugate Gradient.

    Args:
        psf_shape (tuple[int]): PSF footprint
        patch_pair (npt.NDArray[np.float64]): stack of the two patches from each image in the pair
        verbose (bool, optional): whether to show iterative progress bar. Defaults to False.

    Returns:
        npt.NDArray[np.float64]: PSF
    """
    kspace_pair = sp.fft(patch_pair, axes=(0, 1))
    kspace_in, kspace_out = kspace_pair[..., 0], kspace_pair[..., 1]
    A = forward_model(kspace_in, psf_shape)
    # A = forward_model_conv(patch_pair[..., 0], psf_shape)
    y = sp.resize(patch_pair[..., 1], A.oshape)
    # A_mat = get_matrix(A)
    # A_mat = np.random.rand(np.prod(A.oshape), np.prod(A.ishape))
    # A_inv = np.linalg.pinv(A_mat) # direct solve takes about 4 times longer than iterative solve using a random matrix as a proxy for A. If you call get_matrix and actually construct A by brute force it takes 10x instead of 4x.
    # soln = np.reshape(A_inv.dot(y.ravel()), psf_shape)
    app = sp.app.LinearLeastSquares(A, y, x=np.zeros(A.ishape, dtype=np.complex128), tol=1e-10, max_iter=1e10, show_pbar=verbose, lamda=0)
    # app = sp.app.LinearLeastSquares(A, y, x=np.zeros(A.ishape, dtype=np.float64), tol=1e-10, max_iter=1e10, show_pbar=verbose, lamda=0)
    soln = app.run()
    psf = np.abs(soln)
    return psf

def forward_model(
        image_dft: npt.NDArray[np.float64],
        psf_shape: tuple[int]
        ) -> sp.linop.Linop:
    """Linear operator representing convolution of an image with a PSF (input).

    Operator performs a "valid" convolution (no padding or wrapping). 
    Model implements convolution in the DFT domain for computational efficiency.

    Args:
        input_kspace (npt.NDArray[np.float64]): 2D DFT of 3D image patch
        psf_shape (tuple[int]): PSF footprint; expects 3D PSF with singleton 3rd dimension

    Returns:
        sp.linop.Linop: linear operator
    """
    Z = sp.linop.Resize(image_dft.shape[:2] + (1,), psf_shape)
    F = sp.linop.FFT(Z.oshape, axes=(0, 1))
    D = sp.linop.Multiply(F.oshape, image_dft)
    no_wrap_size = tuple(np.array(image_dft.shape[:2]) - np.array(psf_shape[:2]) + np.ones(2, dtype=int))
    FH = sp.linop.FFT(D.oshape, axes=(0, 1)).H
    C = sp.linop.Resize(no_wrap_size + image_dft.shape[2:], FH.oshape)
    return C * FH * D * F * Z

def forward_model_conv(
        input_image: npt.NDArray[np.float64],
        psf_shape: tuple[int]
        ) -> sp.linop.Linop:
    """Alternate implementation of forward model.

    Image-based implementation is much simpler but also less efficient.

    Args:
        input_image (npt.NDArray[np.float64]): 3D image patch
        psf_shape (tuple[int]): PSF footprint

    Returns:
        sp.linop.Linop: linear operator
    """
    A = sp.linop.ConvolveFilter(psf_shape, input_image, mode='valid')
    return A

def get_FWHM_from_image(
        psf: npt.NDArray[np.float64],
        psf_shape: tuple[int],
        num_workers: int,
        stride: int = 1,
        batch_axis: int = 2
        ) -> npt.NDArray[np.float64]:
    """Measure the FWHM for each local PSF estimated from the image.

    Args:
        psf (npt.NDArray[np.float64]): PSF map
        psf_shape (tuple[int]): PSF footprint
        num_workers (int): parallelization factor
        stride (int, optional): stride between patches. Defaults to 1.
        batch_axis (int, optional): axis along which to batch patches for parallel computation. Defaults to 2.

    Returns:
        npt.NDArray[np.float64]: FWHM map
    """
    func = get_FWHM_from_pixel
    patch_shape = (1, 1, 1)
    out_shape = (sum(p > 1 for p in psf_shape),) # number of non-singleton dimensions of PSF
    return generic_filter(psf, func, patch_shape, out_shape, stride, batch_axis, num_batches=num_workers)

def get_FWHM_from_pixel(
        psf: npt.NDArray[np.float64]
        ) -> list[float]:
    """Measure the FWHM of a point spread function (PSF).

    Args:
        psf (npt.NDArray[np.float64]): PSF

    Returns:
        list[float]: FWHM in each dimension
    """
    psf = np.abs(np.squeeze(psf))
    ndim = psf.ndim
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

def get_FWHM_from_vec(
        x: npt.NDArray[np.float64],
        i_max: int = None
        ) -> float:
    """Measure the FWHM of a 1D array. 

    Args:
        x (npt.NDArray[np.float64]): target array
        i_max (int, optional): index of array's maximum value. Defaults to None.

    Returns:
        float: FWHM
    """
    if i_max is None:
        i_max = np.argmax(x)
    half_max = x[i_max] / 2
    if i_max < 1 or i_max >= len(x) - 1:
        return 0
    i_half_max = i_max - np.argmin(x[i_max::-1] > half_max)  # just left of half-max
    if i_half_max == i_max:
        i_half_max_1 = None
    else:
        i_half_max_1 = find_root(i_half_max,
                                 x[i_half_max] - half_max,
                                 i_half_max + 1,
                                 x[i_half_max + 1] - half_max)
    i_half_max = i_max + np.argmin(x[i_max::1] > half_max)  # just right of half-max
    if i_half_max == i_max:
        i_half_max_2 = None
    else:
        i_half_max_2 = find_root(i_half_max - 1,
                                 x[i_half_max - 1] - half_max,
                                 i_half_max,
                                 x[i_half_max] - half_max)
    if i_half_max_1 is None or i_half_max_2 is None:
        print('Warning: PSF half-max extent exceeds window')
    if i_half_max_1 is not None and i_half_max_2 is not None:
        return i_half_max_2 - i_half_max_1
    if i_half_max_1 is None and i_half_max_2 is not None:
        return 2 * (i_half_max_2 - i_max)
    if i_half_max_1 is not None and i_half_max_2 is None:
        return 2 * (i_max - i_half_max_1)
    if i_half_max_1 is None and i_half_max_2 is None:
        return 0

def find_root(x1: int, y1: float, x2: int, y2: float) -> float:
    """ Find zero-crossing of line through points (x1, y1) and (x2, y2). """
    if y1 == y2:
        # TODO add some epsilon for when points are too close together, just return one of them
        print('find_root division by zero')
    return x1 - y1 * (x2 - x1) / (y2 - y1)

def plot_map(
        ax: plt.Axes,
        res_map: npt.NDArray[np.float64],
        mask: npt.NDArray[np.bool],
        vmin: float = 1.2,
        vmax: float = 3.6,
        show_cbar: bool = True
        ) -> mpl.colorbar.Colorbar | None:
    """Plot SR map.

    Args:
        ax (plt.Axes): target plot 
        sr_map (npt.NDArray[np.float64]): SR map
        mask (npt.NDArray[np.bool]): mask identifying areas where map is valid
        vmin (float, optional): lower limit for color range. Defaults to 1.2.
        vmax (float, optional): upper limit for color range. Defaults to 3.6.
        show_cbar (np.bool, optional): whether to include a colorbar. Defaults to True.

    Returns:
        mpl.colorbar.Colorbar: colorbar
    """
    im = ax.imshow(res_map, cmap=CMAP['resolution'], vmin=vmin, vmax=vmax)
    if mask is not None:
        overlay_mask(ax, ~mask)
    if show_cbar:
        cbar = colorbar(ax, im, 'FWHM (mm)', ticks=[vmin, vmin + (vmax-vmin)/2, vmax])
        # cbar = plt.colorbar(im, cax=colorbar_axis(ax), ticks=[vmin, vmin + (vmax-vmin)/2, vmax])
        return cbar

def colorbar(
        ax: plt.Axes,
        im: mpl.image.AxesImage,
        label: str,
        offset: float = 0,
        ticks: list[float] = [1, 2, 3]
        ) -> mpl.colorbar.Colorbar:
    """Plot colorbar for SR map.

    Args:
        ax (plt.Axes): where map is plotted
        im (mpl.image.AxesImage): mappable image from SR plot
        label (str): label for colorbar
        offset (float, optional): positional offset of colorbar from IA map. Defaults to 0.
        ticks (list[float]): colorbar ticks

    Returns:
        mpl.colorbar.Colorbar: colorbar
    """
    cbar = plt.colorbar(im, cax=colorbar_axis(ax, offset=offset), ticks=ticks)
    cbar.set_label(label, size=SMALL_SIZE)
    cbar.ax.tick_params(labelsize=SMALLER_SIZE)
    return cbar

def gaussian_blur(
        image: npt.NDArray[np.float64],
        sigma: float,
        psf_radius: int,
        axes: tuple[int] = (0,)
        ) -> npt.NDArray[np.float64]:
    """ Apply gaussian blurring to image. """
    return ndi.gaussian_filter(image, sigma, order=0, radius=psf_radius, output=None, mode='constant', axes=axes)

def gaussian_psf(
        shape: tuple[int],
        sigma: float,
        psf_radius: int,
        axes: tuple[int] = (0,)
        ) -> npt.NDArray[np.float64]:
    """ Return gaussian PSF. """
    image = unit_impulse(shape, idx='mid')
    return gaussian_blur(image, sigma, psf_radius, axes=axes)
