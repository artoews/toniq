"""Functions for making lattice unit cells.

"""
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from toniq.plot import plotVolumes

def gyroid_unit_cell(size: int, resolution: int) -> npt.NDArray:
    pts = np.arange(0, size, resolution)
    x, y, z = np.meshgrid(pts, pts, pts, indexing='ij')
    # approximate form of level set from https://en.wikipedia.org/wiki/Gyroid
    g = np.sin(2 * np.pi / size * x) * np.cos(2 * np.pi / size * y) + \
        np.sin(2 * np.pi / size * y) * np.cos(2 * np.pi / size * z) + \
        np.sin(2 * np.pi / size * z) * np.cos(2 * np.pi / size * x)
    return g

def cubic_unit_cell(size: int, resolution: int, line_width: int) -> npt.NDArray:
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

def make_lattice(type: str, shape=(1, 1, 1), resolution=1):
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

if __name__ == '__main__':
    lattice = make_lattice('gyroid')
    # lattice = make_lattice('cubic')
    fig1, tracker1 = plotVolumes((lattice,))
    plt.show()
