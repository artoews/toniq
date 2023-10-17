import numpy as np
import sigpy as sp

def resize_image_matrix(image, shape):
    return np.abs(sp.ifft(sp.resize(sp.fft(image), shape)))

def safe_divide(divisor, dividend, thresh=0):
     return np.divide(divisor, dividend, out=np.zeros_like(divisor), where=np.abs(dividend) > thresh)