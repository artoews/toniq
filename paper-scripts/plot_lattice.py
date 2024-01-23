import numpy as np
import matplotlib.pyplot as plt
from os import path

from lattice import make_lattice, get_kspace_center, get_condition
from plot import letter_annotation

from plot_params import *

styles = ['dotted', 'solid', 'dashed']

def plot_image_panel(fig, cubic, gyroid, vmax=1, log=False):
    num_slices = cubic.shape[-1]
    axes = fig.subplots(nrows=2, ncols=num_slices)
    for i in range(num_slices):
        plot_image(axes[0, i], cubic[:, :, i], vmax=vmax, log=log)
        plot_image(axes[1, i], gyroid[:, :, i], vmax=vmax, log=log)
    axes[0, 0].set_ylabel('Cubic')
    axes[1, 0].set_ylabel('Gyroid')
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

def plot_condition(fig, psf_sizes, gyroid_conditions, cubic_conditions):
    ax = fig.subplots()
    ax.plot(psf_sizes, cubic_conditions, label='Cubic', linestyle='dashed')
    ax.plot(psf_sizes, gyroid_conditions, label='Gyroid', linestyle='solid')
    ax.set_xlabel('PSF Size (pixels)')
    ax.set_ylabel('Condition Number')
    ax.legend()
    return ax


if __name__ == '__main__':

    save_dir = '/Users/artoews/root/code/projects/metal-phantom/lattice/'
    num_slices = 10
    patch_shape = (20, 20, 20)
    psf_sizes = range(4, 11)

    psf_shapes = [(size,) * 3 for size in psf_sizes]

    gyroid = make_lattice('gyroid')
    gyroid_k = get_kspace_center(gyroid, patch_shape)
    cubic = make_lattice('cubic')
    cubic_k = get_kspace_center(cubic, patch_shape)

    # print('getting conditions')
    # gyroid_conditions = [get_condition(gyroid_k, shape) for shape in psf_shapes]
    # cubic_conditions = [get_condition(cubic_k, shape) for shape in psf_shapes]

    # np.savez(path.join(save_dir, 'outputs.npz'), gyroid_conditions=gyroid_conditions, cubic_conditions=cubic_conditions, psf_sizes=psf_sizes)
    data = np.load(path.join(save_dir, 'outputs.npz'))
    for var in data:
        globals()[var] = data[var]
    # print('gyroid condition', gyroid_conditions)
    # print('cubic condition', cubic_conditions)

    ## Setup
    fig = plt.figure(figsize=(12, 4), layout='constrained')
    fig_AB, fig_C = fig.subfigures(1, 2, wspace=0.1, width_ratios=(3, 1))
    fig_A, fig_B = fig_AB.subfigures(2, 1, hspace=0.1, height_ratios = (1, 1))

    ## Plot
    step = cubic.shape[-1] // num_slices
    axes_A = plot_image_panel(fig_A, cubic[:, :, ::step], gyroid[:, :, ::step], vmax=2)
    letter_annotation(axes_A[0][0], -0.2, 1.1, 'A')
    start = cubic_k.shape[-1] // 2 - num_slices // 2
    cubic_k_patch = cubic_k[:, :, start:start+num_slices]
    gyroid_k_patch = gyroid_k[:, :, start:start+num_slices]
    axes_B = plot_image_panel(fig_B, cubic_k_patch, gyroid_k_patch, vmax=7, log=True)
    letter_annotation(axes_B[0][0], -0.2, 1.1, 'B')
    axes_C = plot_condition(fig_C, psf_sizes, gyroid_conditions, cubic_conditions)
    letter_annotation(axes_C, -0.2, 1.1, 'C')

    ## Save
    plt.savefig(path.join(save_dir, 'condition_analysis.png'), dpi=300)
    plt.show()