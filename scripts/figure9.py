import argparse
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import sigpy as sp
import yaml

from os import path, makedirs
from matplotlib.ticker import MultipleLocator

import snr, sr
from plot import plotVolumes
from scipy.signal import unit_impulse

from plot import color_panels, label_panels, remove_ticks
from plot_params import *

from config import parse_slice
from masks import get_implant_mask, get_signal_mask
from util import normalize, load_series_from_path

patch_shape = (14, 14, 10)
psf_radius = 20

def gaussian_blur(image, sigma, axes=(0,)):
    return ndi.gaussian_filter(image, sigma, order=0, radius=psf_radius, output=None, mode='constant', axes=axes)

def gaussian_psf(shape, sigma, axes=(0,)):
    image = unit_impulse(shape, idx='mid')
    return gaussian_blur(image, sigma, axes)

def plot_fwhm_maps(maps):
    volumes = []
    for i in range(2):
        for j in range(len(maps)):
            volumes += [maps[j][..., i]]
    fig, tracker = plotVolumes(volumes, figsize=(12, 4), nrows=2, ncols=len(maps), vmin=1, vmax=3, cmap=CMAP['resolution'], cbar=True)
    return fig, tracker

def plot_distribution(ax, targets, maps, major_grid=True, minor_grid=True):
    maps = [maps[i][maps[i]>0] for i in range(len(maps))]
    ax.violinplot(maps, showmedians=True)
    ax.set_xlabel('Retrospective Trials')
    # ax.set_xticks(range(1, len(maps)+1))
    ax.set_xticks([])
    ax.set_ylabel('FWHM (pixels)')
    ax.set_ylim([0.75, 3.25])
    ax.set_yticks(np.round(targets, 2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.125))
    if major_grid:
        ax.grid(axis='y', which='major', linestyle='solid')
    if minor_grid:
        ax.grid(axis='y', which='minor', linestyle='dotted')
    return fig, axes

def plot_row(axes, images, slc=None, shape=None, cmap=CMAP['image'], vmin=0, vmax=1, normalize=False):
    ims = []
    for ax, image in zip(axes, images):
        if slc is not None:
            image = image[slc]
        if shape is not None:
            image = sp.resize(np.squeeze(image), shape)
        if normalize:
            image = image / np.max(np.abs(image))
        im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        ims += [im]
    return ims

p = argparse.ArgumentParser(description='Make Figure 9')
p.add_argument('save_dir', type=str, help='path where outputs are saved')
p.add_argument('config', type=str, default=None, help='yaml config file specifying data paths and mapping parameters')
p.add_argument('-n', '--noise', type=float, default=0.01, help='standard deviation of noise added to normalized image; default=0.01')
p.add_argument('-l', '--load', action='store_true', help='load inputs from save_dir')
p.add_argument('-p', '--plot', action='store_true', help='show plots')
p.add_argument('-x', type=int, default=100, help='x coordinate of inset location')
p.add_argument('-y', type=int, default=92, help='x coordinate of inset location')
p.add_argument('-w', '--window_size', type=int, default=14, help='window size in pixels; default=14')
p.add_argument('--psf_size', type=int, default=5, help='psf size in pixels; default=5')
p.add_argument('-s', '--sigma', type=float, nargs='+', default=[0.3, 0.558, 0.675, 0.769, 0.85, 0.926, 1.02, 1.128, 1.247], help='sigmas for gaussian PSF; default = sigmas yielding FWHM=1:3:0.25 pixels')


if __name__ == '__main__':

    # process args
    args = p.parse_args()
    if not path.exists(args.save_dir):
        makedirs(args.save_dir)

    # process config
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    slc = parse_slice(config)

    if not args.load:

        # load reference image
        series_path = config['dicom-series']['structured-plastic-reference']
        image = load_series_from_path(series_path)
        # resolution_mm = image.meta.resolution_mm
        resolution_mm = [1.2, 1.2, 1.2]
        reference_image = image.data
        reference_image = np.abs(sp.ifft(sp.resize(sp.fft(reference_image), (256, 256, 64))))
        reference_image = normalize(reference_image)
        print('loaded reference image with shape', reference_image.shape)

        # generate blurred targets and associated PSFs
        target_images = []
        target_psfs = []
        for sigma in args.sigma:
            target_images += [gaussian_blur(reference_image, sigma)]
            target_psfs += [gaussian_psf(reference_image.shape, sigma)]

        # reduce image to target slice of interest
        reference_image = reference_image[slc]
        target_images = [image[slc] for image in target_images]
        target_psfs = [sp.resize(psf, (psf_radius, psf_radius, 1)) for psf in target_psfs]

        # measured correct FWHM
        measured_fwhms = []
        for target_psf in target_psfs:
            measured_fwhm = sr.get_FWHM_from_pixel(target_psf)
            measured_fwhms += [measured_fwhm[0]]
        # print('Measured FWHM', measured_fwhms)
        # quit()
    
        # add noise
        target_images_2 = []
        if args.noise != 0:
            np.random.seed(0)
            for i in range(len(target_images)):
                noise2 = np.random.normal(size=target_images[i].shape, scale=args.noise)
                target_images_2 += [target_images[i] + noise2]
                noise = np.random.normal(size=target_images[i].shape, scale=args.noise)
                target_images[i] += noise
        

        # get mask
        implant_mask = get_implant_mask(reference_image)
        mask = get_signal_mask(implant_mask)

        # check SNR
        snr_maps = []
        for i in range(len(target_images)):
            snr, _, _ = snr.get_map(target_images[i], target_images_2[i], mask)
            snr_maps += [snr]
            print('Trial {} has SNR stats (min {}, mean {}, median {}, max {}): '.format(i, np.min(snr), np.mean(snr), np.median(snr), np.max(snr)))

        # map resolution
        psf_maps = []
        fwhm_maps = []
        stride = config['params']['psf-stride']
        num_workers = config['params']['num-workers']
        for i in range(len(target_images)):
            print('Mapping resolution for case {} of {} with sigma {}'.format(i+1, len(target_images), args.sigma[i]))
            psf_map, fwhm_map = sr.get_map(reference_image, target_images[i], patch_shape, resolution_mm, mask, stride, num_workers=num_workers)
            psf_maps.append(psf_map)
            fwhm_maps.append(fwhm_map) 
        psf_maps = np.stack(psf_maps)
        fwhm_maps = np.stack(fwhm_maps)

        np.savez(path.join(args.save_dir, 'fig9_outputs.npz'),
            measured_fwhms=measured_fwhms,
            fwhm_maps=fwhm_maps,
            psf_maps=psf_maps,
            target_psfs=target_psfs,
            resolution_mm=resolution_mm,
            reference_image=reference_image,
            target_images=target_images
            )
    
    else:
        data = np.load(path.join(args.save_dir, 'fig9_outputs.npz'))
        for var in data:
            globals()[var] = data[var]
    
    # volumes = [reference_image] + target_images
    # fig1, tracker1 = plotVolumes(volumes)
    # fig2, tracker2 = plotVolumes(target_psfs)
    # fig, tracker = plot_fwhm_maps(fwhm_maps / resolution_mm[0])

    fig = plt.figure(figsize=(FIG_WIDTH[2], FIG_WIDTH[2] * 0.8))
    subfigs = fig.subfigures(2, 1, hspace=0.03, height_ratios=[1.8, 1])
    subfigs[0].suptitle('Retrospective Trials')

    axes = subfigs[0].subplots(nrows=4, ncols=len(args.sigma), gridspec_kw={'wspace': 0, 'hspace': 0, 'bottom': 0.05, 'right': 0.94})
    inset = (slice(args.x, args.x + args.window_size),
             slice(args.y, args.y + args.window_size),
             (slc[2].stop - slc[2].start)//2+10)
    plot_row(axes[0, :], target_psfs, shape=(args.psf_size, args.psf_size), normalize=True)
    plot_row(axes[1, :], target_images, slc=inset)
    plot_row(axes[2, :], psf_maps, slc=(5, 5, 18), normalize=True)
    ims = plot_row(axes[3, :], fwhm_maps / resolution_mm[0], slc=(slice(None), slice(None), 18, 0), vmin=0.75, vmax=3.25)
    for ax, label in zip(axes[:, 0], ('Retrospective\nPSF', 'Target\nPatch', 'Local PSF\nEstimate', 'Up/Down\nResolution\nMap')):
        ax.set_ylabel(label, rotation='horizontal', va='center', ha='center', labelpad=30)
    for ax, title in zip(axes[0, 1:], ['{:.2f}'.format(i) for i in np.arange(1.25, 3.1, 0.25)]):
        ax.set_title(title)
    axes[0, 0].set_title('FWHM =\n1.00')
    remove_ticks(axes)
    sr.colorbar(axes[3, -1], ims[-1], 'FWHM (pixels)')

    axes = subfigs[1].subplots(nrows=1, ncols=2, gridspec_kw={'bottom': 0.15, 'top': 0.85, 'right': 0.94})
    plot_distribution(axes[0], measured_fwhms, [maps[..., 0] / resolution_mm[0] for maps in fwhm_maps])
    plot_distribution(axes[1], (1.00,), [maps[..., 1] / resolution_mm[1] for maps in fwhm_maps])
    axes[0].set_title('Up/Down Resolution')
    axes[1].set_title('Left/Right Resolution')

    color_panels(subfigs.flat)
    label_panels(subfigs.flat)

    plt.savefig(path.join(args.save_dir, 'figure9.png'), dpi=DPI)

    if args.plot:
        plt.show()
