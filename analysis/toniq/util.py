import dicom
import inspect
import json
import numpy as np
import sigpy as sp
import sys

from os import path
from pathlib import Path
from skimage import filters, morphology
from time import time

def load_series(exam_root, series_name):
    series_path = path.join(exam_root, series_name)
    files = Path(series_path).glob('*MRDC*')
    image = dicom.load_series(files)
    return image

def load_series_from_path(series_path):
    files = Path(series_path).glob('*MRDC*')
    image = dicom.load_series(files)
    return image

def equalize(images, pct=99, axis=0):
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

def normalize(image, pct=99):
    return image / np.percentile(np.abs(image), pct)

def resize_image_matrix(image, shape):
    return np.abs(sp.ifft(sp.resize(sp.fft(image), shape)))

def safe_divide(divisor, dividend, thresh=0):
     return np.divide(divisor, dividend, out=np.zeros_like(divisor), where=np.abs(dividend) > thresh)
    
def debug(msg, start_time=None):
    frame = inspect.stack()[1][0]
    info = inspect.getframeinfo(frame)
    file = path.basename(info.filename)
    func = info.function
    lineno = info.lineno
    msg = '{}::{}()::{} {}.'.format(file, func, lineno, msg)
    if start_time is not None:
        msg += ' {:.0f} seconds elapsed.'.format(time() - start_time)
    print(msg)

def coord_mats(shape, res=None, loc=(0.5, 0.5, 0.5), offset=0):
    if res is None:
        res = (1,) * len(shape)
    coord_vecs = (r * (np.arange(s, dtype=float) - int(s * l) + offset) for r, s, l in zip(res, shape, loc))
    return np.meshgrid(*coord_vecs, indexing='ij')

def masked_copy(arr, mask, fill_val=0):
    ''' return copy of arr with values outside mask set to fill_val '''
    arr_copy = arr.copy()
    arr_copy[~mask] = fill_val
    return arr_copy

def save_args(args, save_dir):
    args_dict = args.__dict__
    args_dict['cmd'] = " ".join(["python"] + sys.argv)
    with open(path.join(save_dir, 'args.txt'), 'w') as f:
        json.dump(args_dict, f, indent=4)

def list_to_formatted_string(x, format='int'):
    if format == 'int':
        format_list = ['{:.3g}' for _ in x]
    s = ', '.join(format_list)
    return s.format(*x)