import numpy as np
import matplotlib.pyplot as plt
import sigpy as sp

from toniq.linop import get_matrix
from toniq.plot import plotVolumes
from toniq.sr import forward_model

def gyroid_unit_cell(size, resolution):
    pts = np.arange(0, size, resolution)
    x, y, z = np.meshgrid(pts, pts, pts, indexing='ij')
    # approximate form of level set from https://en.wikipedia.org/wiki/Gyroid
    g = np.sin(2 * np.pi / size * x) * np.cos(2 * np.pi / size * y) + \
        np.sin(2 * np.pi / size * y) * np.cos(2 * np.pi / size * z) + \
        np.sin(2 * np.pi / size * z) * np.cos(2 * np.pi / size * x)
    return g

def cubic_unit_cell(size, resolution, line_width):
    pts = np.arange(0, size, resolution)
    x, y, z = np.meshgrid(pts, pts, pts, indexing='ij')
    g = (np.mod(x, size / 2) < line_width).astype(np.int) + \
        (np.mod(y, size / 2) < line_width).astype(np.int) + \
        (np.mod(z, size / 2) < line_width).astype(np.int)
    cell = g > 1
    half_width = int(line_width / resolution / 2)
    cell = np.roll(cell, -half_width, axis=0)
    cell = np.roll(cell, -half_width, axis=1)
    cell = np.roll(cell, -half_width, axis=2)
    return cell

def make_lattice(type, shape=(1, 1, 1), resolution=1):
    size = 120
    line_width = 10
    # size = size / 5
    # line_width = line_width / 5
    if type == 'gyroid':
        cell_surface = gyroid_unit_cell(size, resolution)
        cell_solid = np.abs(cell_surface) < line_width / 28 # found empirically to give the prescribed line_width 
    elif type == 'cubic':
        cell_solid = cubic_unit_cell(size, resolution, line_width)
    lattice = np.tile(1-cell_solid, shape)
    return lattice

def get_condition(kspace, psf_shape, lamda=0):
    A_op = forward_model(kspace, psf_shape)
    # A_op = forward_model_conv(kspace, psf_shape)
    # print(A_op)
    A_mtx = get_matrix(A_op, verify=True)
    # print(A_mtx.shape)
    if lamda != 0:
        A_mtx = np.vstack((A_mtx, np.eye(A_mtx.shape[-1]) * np.sqrt(lamda)))
    return np.linalg.cond(A_mtx)

def get_kspace_center(lattice, init_shape, final_shape=None):
    kspace = sp.ifft(sp.resize(sp.fft(lattice), init_shape), axes=(2,))
    if final_shape is not None:
        kspace = sp.fft(sp.resize(sp.ifft(kspace, axes=(0, 1)), final_shape), axes=(0, 1))
    # kspace = sp.ifft(sp.resize(sp.fft(lattice), shape), axes=(0, 1, 2)) # temp fix for conv model
    return kspace


if __name__ == '__main__':
    lattice = make_lattice('gyroid')
    # lattice = make_lattice('cubic')
    fig1, tracker1 = plotVolumes((lattice,))
    plt.show()
