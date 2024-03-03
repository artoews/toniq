import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import sigpy as sp
import yaml

from os import path, makedirs
from pathlib import Path

from retro_res import gaussian_blur, gaussian_psf
from slice_params import *
from plot_params import *
from plot import remove_ticks, color_panels, label_panels
from util import load_series_from_path, normalize

kwargs = {'vmin': 0, 'vmax': 1, 'cmap': CMAP['image']}

def parse_slice(config):
    return tuple(slice(start, stop) for start, stop in config['params']['slice'])

def plot_inputs(fig, target, reference, inset):
    axes = fig.subplots(2, 1, gridspec_kw={'left': 0.05, 'right': 0.95, 'bottom': 0.05, 'top': 0.92})
    axes[0].imshow(reference, **kwargs)
    axes[1].imshow(target, **kwargs)
    axes[0].set_title('Reference')
    axes[1].set_title('Target')
    remove_ticks(axes)
    inset_start = (inset[0].start, inset[1].start)
    inset_size = (inset[0].stop - inset[0].start, inset[1].stop - inset[1].start)
    for ax in axes.flat:
        rect = patches.Rectangle(inset_start, *inset_size, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

def plot_model(fig, target, reference, psf):
    axes = fig.subplots(2, 3, gridspec_kw={'left': 0.05, 'right': 0.95, 'bottom': 0.05, 'top': 0.92})
    titles = ('Target', 'Reference', 'Local PSF')
    images = (target, reference, psf)
    for i in range(3):
        axes[0, i].imshow(images[i], **kwargs)
        axes[0, i].set_title(titles[i])
        kspace = np.abs(sp.fft(images[i]))
        kspace = np.log(kspace + 1) * 2
        axes[1, i].imshow(kspace, **kwargs)
    axes[0, 0].set_ylabel('Patch')
    axes[1, 0].set_ylabel('2D DFT')
    symbols = ('=', r'$\circledast$', '=', r'$\odot$')
    for ax, symbol in zip(axes[:, :2].flat, symbols):
        ax.text(1.1, 0.49, symbol, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, size=20)
    remove_ticks(axes)

p = argparse.ArgumentParser(description='Make figure 4')
p.add_argument('root', type=str, help='path where outputs are saved')
p.add_argument('config', type=str, default=None, help='yaml config file specifying data paths and mapping parameters')
p.add_argument('-x', type=int, default=100, help='x coordinate of inset location')
p.add_argument('-y', type=int, default=92, help='x coordinate of inset location')
p.add_argument('-w', '--window_size', type=int, default=10, help='window size in pixels')
p.add_argument('-p', '--psf_size', type=int, default=5, help='psf size in pixels')
p.add_argument('-s', '--sigma', type=float, nargs='+', default=[1, 0.5], help='sigmas for gaussian PSF')

if __name__ == '__main__':

    # process args
    args = p.parse_args()
    save_dir = path.join(args.root, Path(args.config).stem)
    if not path.exists(save_dir):
        makedirs(save_dir)

    # process config
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    slc = parse_slice(config)
    slc = (slc[0], slc[1], (slc[2].stop - slc[2].start)//2+10)

    inset = (slice(args.x, args.x + args.window_size),
             slice(args.y, args.y + args.window_size))

    series_path = config['dicom-series']['structured-plastic-reference']
    image = load_series_from_path(series_path)
    resolution_mm = image.meta.resolution_mm
    reference = image.data
    reference = np.abs(sp.ifft(sp.resize(sp.fft(reference), (256, 256, 64))))
    reference = normalize(reference[slc], pct=100)
    target = gaussian_blur(reference, args.sigma, axes=(0, 1))
    psf = gaussian_psf(reference.shape, args.sigma, axes=(0, 1))
    psf = sp.resize(psf, (args.window_size, args.window_size))
    psf = psf / np.max(psf)

    fig = plt.figure(figsize=(FIG_WIDTH[2], FIG_WIDTH[2]*0.5))
    subfigs = fig.subfigures(1, 2, width_ratios=[1, 3], wspace=0.04)

    plot_inputs(subfigs[0], target, reference, inset)
    plot_model(subfigs[1], target[inset], reference[inset], psf)

    label_panels(subfigs)
    color_panels(subfigs)

    plt.savefig(path.join(args.root, 'figure4.png'), dpi=DPI)

    plt.show()