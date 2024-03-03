import numpy as np
import matplotlib.pyplot as plt
from os import path

from lattice import make_lattice, get_kspace_center, get_condition
from plot import letter_annotation

from plot_params import *

styles = ['dotted', 'solid', 'dashed']

def plot_cell(ax, vol, vmax=1):
    filled = np.ones_like(vol)
    filled[1:-1, 1:-1, 1:-1] = 0
    facecolors = np.empty(vol.shape + (3,), dtype=np.float)
    facecolors[vol==0] = np.zeros(3)
    facecolors[vol==1] = np.ones(3) / vmax
    ax.voxels(filled, facecolors=facecolors, edgecolors=facecolors)
    ax.set(xlabel='x', ylabel='y', zlabel='z')
    ax.set_aspect('equal')
    ax.set_axis_off()
    ax.patch.set_alpha(0)
    return ax

def plot_cells(fig, cubic, gyroid):
    axes = fig.subplots(nrows=2, subplot_kw={'projection': '3d'})
    # ax = fig.add_subplot(projection='3d')
    plot_cell(axes[0], cubic)
    plot_cell(axes[1], gyroid)
    return axes

def plot_image_panel(fig, cubic, gyroid, vmax=1, log=False):
    num_slices = cubic.shape[-1]
    axes = fig.subplots(nrows=2, ncols=num_slices)
    for i in range(num_slices):
        plot_image(axes[0, i], cubic[:, :, i], vmax=vmax, log=log)
        plot_image(axes[1, i], gyroid[:, :, i], vmax=vmax, log=log)
    # axes[0, 0].set_ylabel('Cube')
    # axes[1, 0].set_ylabel('Gyroid')
    return axes

def plot_image(ax, image, xlabel=None, ylabel=None, title=None, vmax=1, log=False):
    image = np.abs(image)
    if log:
        image = np.log(image+1)
    im = ax.imshow(image, vmin=0, vmax=vmax, cmap=CMAP['image'])
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.set_xticks([])
    ax.set_yticks([])
    if title is not None:
        ax.set_title(title)
    return im

def plot_condition(fig, l2_list, psf_sizes, cubic, gyroid):
    cubic_colors = plt.cm.Greys(np.linspace(0.25, 0.75, len(l2_list)))
    gyroid_colors = plt.cm.Greens(np.linspace(0.25, 0.75, len(l2_list)))
    styles = ['dotted', 'dashed', 'solid']
    ax = fig.subplots()
    for i in range(len(l2_list)):
        # ax.plot(psf_sizes, cubic[i], label=r'Cube, $\lambda={}$'.format(l2_list[i]), linestyle='dashed', color=cubic_colors[i])
        ax.plot(psf_sizes, cubic[i], label=r'Cube, $\lambda={}$'.format(l2_list[i]), linestyle=styles[i], color='grey')
    for i in range(len(l2_list)):
        ax.plot(psf_sizes, gyroid[i], label=r'Gyroid, $\lambda={}$'.format(l2_list[i]), linestyle=styles[i], color='green')
    ax.set_xlabel('PSF Size (pixels)')
    ax.set_ylabel('Condition Number')
    # ax.set_yscale('log')
    # ax.set_ylim([1e0, 1e4])
    ax.legend()
    ax.set_xlim([min(psf_sizes), max(psf_sizes)])
    ax.set_ylim([0, 300])
    plt.grid()
    return ax


if __name__ == '__main__':

    save_dir = '/Users/artoews/root/code/projects/metal-phantom/lattice-feb27/'
    res = 1  # should be 1, but can increase to save time
    load_cond = False
    l2_list = [0,]
    patch_shape = (10, 10, 10) # remember the cell shape is in integer steps of unit cell. so don't set this to 14!!
    psf_sizes = range(3, 8)
    psf_shapes = [(size, size, 1) for size in psf_sizes]
    
    # TODO update to accomodate non-integer cell sizes, i.e. cropping in spatial domain

    cell_shape = (1, 1, 1)
    gyroid = make_lattice('gyroid', resolution=res, shape=cell_shape)
    gyroid_k = get_kspace_center(gyroid, patch_shape)
    cubic = make_lattice('cubic', resolution=res, shape=cell_shape)
    cubic_k = get_kspace_center(cubic, patch_shape)

    if load_cond:
        data = np.load(path.join(save_dir, 'outputs.npz'))
        for var in data:
            globals()[var] = data[var]
    else:
        condition_cubic = []
        condition_gyroid = []
        for l2 in l2_list:
            print('l2={}'.format(l2))
            condition_cubic.append([get_condition(cubic_k, shape, lamda=l2) for shape in psf_shapes])
            condition_gyroid.append([get_condition(gyroid_k, shape, lamda=l2) for shape in psf_shapes])
        np.savez(path.join(save_dir, 'outputs.npz'), condition_cubic=condition_cubic, condition_gyroid=condition_gyroid, psf_sizes=psf_sizes)

    print('cubic condition', condition_cubic)
    print('gyroid condition', condition_gyroid)

    # fig = plt.figure(figsize=(12, 4), layout='constrained')
    # fig_A, fig_B, fig_C = fig.subfigures(1, 3, wspace=0.1, width_ratios=(2, 5, 3))
    # axes_A = plot_cells(fig_A, cubic, gyroid)
    # start = cubic_k.shape[-1] // 2
    # axes_B = plot_image_panel(fig_B, cubic_k[:, :, :3], gyroid_k[:, :, :3], vmax=7, log=True)
    # axes_C = plot_condition(fig_C, l2_list, psf_sizes, condition_cubic, condition_gyroid)
    # letter_annotation(axes_A[0], -0.2, 1.1, 'A')
    # letter_annotation(axes_B[0][0], -0.2, 1.1, 'B')
    # letter_annotation(axes_C, -0.2, 1.1, 'C')

    fig = plt.figure(figsize=(5, 5), layout='constrained')
    axes = plot_condition(fig, l2_list, psf_sizes, condition_cubic, condition_gyroid)
    plt.savefig(path.join(save_dir, 'condition_analysis.png'), dpi=300)
    plt.show()