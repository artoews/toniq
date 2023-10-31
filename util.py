import inspect
import numpy as np
import sigpy as sp

from os import path
from time import time

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
