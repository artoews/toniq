import numpy as np
import matplotlib.pyplot as plt
from linop import get_matrix
from plot import plotVolumes
from resolution import forward_model
import sigpy as sp
import scipy.ndimage as ndi
from skimage import morphology
from sklearn.neighbors import NearestNeighbors

def gyroid_unit_cell(size, resolution):
    pts = np.arange(0, size, resolution)
    x, y, z = np.meshgrid(pts, pts, pts, indexing='ij')
    # approximate form of level set from https://en.wikipedia.org/wiki/Gyroid
    g = np.sin(2 * np.pi / size * x) * np.cos(2 * np.pi / size * y) + \
        np.sin(2 * np.pi / size * y) * np.cos(2 * np.pi / size * z) + \
        np.sin(2 * np.pi / size * z) * np.cos(2 * np.pi / size * x)
    return g

def cubic_unit_cell(size, resolution, line_width, thresh=0.99):
    pts = np.arange(0, size, resolution)
    x, y, z = np.meshgrid(pts, pts, pts, indexing='ij')
    g = (np.mod(x + resolution, size / 2) < line_width * resolution) + \
        (np.mod(y + resolution, size / 2) < line_width * resolution) + \
        (np.mod(z + resolution, size / 2) < line_width * resolution)
    return g > 0

def largest_hole(mask):
    radius = 0
    prev_hole_mask = mask
    while True:
        radius += 1
        footprint = morphology.ball(radius)
        hole_mask = morphology.binary_dilation(mask, footprint=footprint)
        if hole_mask.all():
            break
        else:
            prev_hole_mask = hole_mask
    return radius, prev_hole_mask

def largest_hole_2(mask):
    nbrs = NearestNeighbors(n_neighbors=1).fit(np.argwhere(mask))
    distances, indices = nbrs.kneighbors(np.argwhere(~mask))
    hole_index = np.argmax(distances)
    return distances[hole_index], indices[hole_index]

def min_of_max(arr, filter_radius):
    footprint = morphology.cube(filter_radius)
    max_arr = ndi.maximum_filter(arr, footprint=footprint, mode='nearest')
    return np.min(max_arr)

if __name__ == '__main__':
    size = 120
    resolution = 1
    line_width = 6
    lattice_shape = (1,) * 3
    patch_shape = (20, 20, 10)

    # psf_shape = (3, 3, 5)
    # psf_shape = (5, 5, 5)
    psf_shape = (10, 10, 5)
    gyroid = False
    if gyroid:
        cell_surface = gyroid_unit_cell(size, resolution)
        cell_solid = np.abs(cell_surface) < line_width / 24 # 24 found empirically to give the prescribed line_width  
    else:
        cell_solid = cubic_unit_cell(size, resolution, line_width)
    lattice = np.tile(cell_solid, lattice_shape)

    fig1, tracker1 = plotVolumes((lattice,))

    print('lattice shape', lattice.shape)

    kspace = sp.fft(lattice)
    kspace = sp.resize(kspace, patch_shape)
    lattice = np.abs(sp.ifft(kspace))
    lattice = lattice / np.max(np.abs(lattice))

    A = forward_model(kspace, psf_shape)
    print('A op shape', A.oshape, A.ishape)
    A_mtx = get_matrix(A, verify=True)
    print('A mtx shape', A_mtx.shape)
    c = np.linalg.cond(A_mtx)
    print('condition number', c)
    # quit()

    origin = np.unravel_index(np.argmax(np.abs(kspace)), kspace.shape)
    origin_val = kspace[origin]
    kspace[origin] = 0
    next_max = np.max(np.abs(kspace))
    # mask = np.abs(kspace) > 0.1 * next_max
    kspace[origin] = origin_val
    # mask = np.abs(kspace) < 0.1 * next_max
    # mask = np.abs(kspace) / np.max(np.abs(kspace)) > 0.02
    # print(np.max(np.abs(kspace)) * 0.2)
    mask = np.abs(kspace) > 10
    filtered_kspace = kspace.copy()
    filtered_kspace[~mask] = 0
    filtered_image = np.abs(sp.ifft(filtered_kspace))
    # print(np.sum(mask))

    # hole_size, hole_mask = largest_hole(mask[24:36, 24:36, 24:36])
    # hole_size, hole_index = largest_hole_2(mask[20:40, 20:40, 20:40])
    # hole_size, hole_index = largest_hole_2(mask)
    # score = min_of_max(np.abs(kspace), 40)
    # score = condition(cell_solid, (10, 10, 10))
    # print(score)

    fig2, tracker2 = plotVolumes((lattice, np.abs(kspace) / next_max, mask, np.abs(filtered_kspace) / next_max, filtered_image))
    fig4, tracker4 = plotVolumes((lattice, np.abs(kspace) / 20, mask))

    fig3 = plt.figure()
    ax3 = fig3.add_subplot(projection='3d')
    ax3.scatter(*np.nonzero(mask))
    plt.show()