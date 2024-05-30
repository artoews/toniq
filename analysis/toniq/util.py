"""Miscellaneous utility functions.

"""
import argparse
import inspect
import json
import numpy.typing as npt
import numpy as np
import sigpy as sp
import sys

from os import path
from skimage import filters, morphology
from time import time

def equalize(
        images: npt.NDArray | list[npt.NDArray],
        pct: float = 99,
        axis: int = 0
        ) -> npt.NDArray | list[npt.NDArray]:
    """ Rescale the intensity levels for a stack of images so that the foreground intensities are synchronized.

    The median intensity in the foreground of each image is synchronized with the first image in the stack. 
    Image foreground is determined by taking all pixels above the Otsu threshold.

    Args:
        images (npt.NDArray | list[npt.NDArray]): stack of images
        pct (float, optional): percentile used to normalize the first image in the stack. Defaults to 99.
        axis (int, optional): axis along which images are stacked, if not already in list format. Defaults to 0.

    Returns:
        npt.NDArray | list[npt.NDArray]: equalized image stack
    """
    if type(images) == np.ndarray:
        images = list(np.moveaxis(images, axis, 0))
        is_array = True
    else:
        is_array = False
    images = [np.abs(image) for image in images]
    images[0] = normalize(images[0], pct=pct)
    otsu_thresholds = [filters.threshold_otsu(image) for image in images]
    signal_masks = [image > thresh for image, thresh in zip(images, otsu_thresholds)]
    signal_masks = [morphology.binary_erosion(mask, footprint=np.ones((4,)*images[0].ndim)) for mask in signal_masks]
    for i in range(1, len(images)):
        images[i] *= np.median(images[0][signal_masks[0]]) / np.median(images[i][signal_masks[i]])
        # images[i] *= np.median(images[0][signal_masks[i]]) / np.median(images[i][signal_masks[i]])
    if is_array:
        images = np.moveaxis(np.stack(images), 0, axis)
    return images

def normalize(
        image: npt.NDArray,
        pct: float = 99
        ) -> npt.NDArray:
    """ Normalize image pixels by the value of a given percentile. """
    return image / np.percentile(np.abs(image), pct)

def resize_image_matrix(
        image: npt.NDArray,
        shape: tuple[int]
        ) -> npt.NDArray:
    """ Resize image by sinc interpolation. """
    return np.abs(sp.ifft(sp.resize(sp.fft(image), shape)))

def safe_divide(
        divisor: npt.NDArray,
        dividend: npt.NDArray,
        thresh: int = 0
        ) -> npt.NDArray:
     """ Divide two arrays, replacing cases of divide-by-zero (or within some threshold of zero) with zeroes. """
     return np.divide(divisor, dividend, out=np.zeros_like(divisor), where=np.abs(dividend) > thresh)
    
def debug(
        msg: str,
        start_time: float = None
        ) -> None:
    """ Debug print statement including details about the call location and (optionally) time elapsed. """
    frame = inspect.stack()[1][0]
    info = inspect.getframeinfo(frame)
    file = path.basename(info.filename)
    func = info.function
    lineno = info.lineno
    msg = '{}::{}()::{} {}.'.format(file, func, lineno, msg)
    if start_time is not None:
        msg += ' {:.0f} seconds elapsed.'.format(time() - start_time)
    print(msg)

def masked_copy(
        arr: npt.NDArray,
        mask: npt.NDArray[np.bool],
        fill_val = 0):
    """ Return copy of array with values outside mask set to the fill value. """
    arr_copy = arr.copy()
    arr_copy[~mask] = fill_val
    return arr_copy

def save_args(
        args: argparse.Namespace,
        save_dir: str
        ) -> None:
    """ Save command-line arguments to a JSON file. """
    args_dict = args.__dict__
    args_dict['cmd'] = " ".join(["python"] + sys.argv)
    with open(path.join(save_dir, 'args.txt'), 'w') as f:
        json.dump(args_dict, f, indent=4)

def list_to_formatted_string(
        x: list,
        format: str = 'int'
        ) -> None:
    """ Convert list to a formatted string. """
    if format == 'int':
        format_list = ['{:.3g}' for _ in x]
    s = ', '.join(format_list)
    return s.format(*x)
