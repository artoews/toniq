import argparse
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import sigpy as sp
import yaml

from os import path, makedirs
from pathlib import Path
from plot import plotVolumes
from scipy.signal import unit_impulse

from plot_params import *

from masks import get_implant_mask, get_signal_mask
from resolution import map_resolution, get_FWHM_from_pixel
from util import normalize, load_series_from_path

patch_shape = (14, 14, 10)
psf_radius = 20

def gaussian_blur(image, sigma, axes=(0,)):
    return ndi.gaussian_filter(image, sigma, order=0, radius=psf_radius, output=None, mode='constant', axes=axes)

def gaussian_psf(shape, sigma, axes=(0,)):
    image = unit_impulse(shape, idx='mid')
    return gaussian_blur(image, sigma, axes)

def parse_slice(config):
    return tuple(slice(start, stop) for start, stop in config['params']['slice'])

def plot_fwhm_maps(maps):
    volumes = []
    for i in range(2):
        for j in range(len(maps)):
            volumes += [maps[j][..., i]]
    fig, tracker = plotVolumes(volumes, figsize=(12, 4), nrows=2, ncols=len(maps), vmin=1, vmax=3, cmap=CMAP['resolution'], cbar=True)
    return fig, tracker

def plot_distribution(fwhm_maps, fwhm_targets):
    fig, axes = plt.subplots(nrows=2, ncols=1)
    num_trials = len(fwhm_maps)
    fwhm_x= [fwhm_maps[i][..., 0][fwhm_maps[i][..., 0] > 0] for i in range(num_trials)]
    fwhm_y= [fwhm_maps[i][..., 1][fwhm_maps[i][..., 1] > 0] for i in range(num_trials)]
    axes[0].violinplot(fwhm_x, showmedians=True)
    axes[1].violinplot(fwhm_y, showmedians=True)
    axes[0].set_yticks(fwhm_targets)
    y_lim = [0, int(np.round(np.max(fwhm_targets)+1))]
    axes[0].set_ylim(y_lim)
    axes[0].grid(axis='y')
    axes[1].set_yticks([1])
    axes[1].set_ylim(y_lim)
    axes[1].grid(axis='y')
    return fig, axes


p = argparse.ArgumentParser(description='Run all four mapping analyses on a single sequence')
p.add_argument('root', type=str, help='path where outputs are saved')
p.add_argument('config', type=str, default=None, help='yaml config file specifying data paths and mapping parameters')
p.add_argument('-n', '--noise', type=float, default=0.01, help='standard deviation of noise added to normalized image; default=0.01')

# sigmas = np.arange(0.45, 1.06, 0.2)
# sigmas =[0.3, 0.675, 0.85, 1.02] # yielding FWHM = [1, 1.5, 2, 2.5]
sigmas =[0.3, 0.558, 0.675, 0.769, 0.85, 0.926, 1.02] # yielding FWHM = [1, 1.5, 2, 2.5]

if __name__ == '__main__':

    # process args
    args = p.parse_args()
    save_dir = path.join(args.root, Path(args.config).stem)
    if not path.exists(save_dir):
        makedirs(save_dir)

    # process config
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    with open(path.join(save_dir, 'config.yml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    slc = parse_slice(config)

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
    for sigma in sigmas:
        target_images += [gaussian_blur(reference_image, sigma)]
        target_psfs += [gaussian_psf(reference_image.shape, sigma)]

    # reduce image to target slice of interest
    reference_image = reference_image[slc]
    target_images = [image[slc] for image in target_images]
    target_psfs = [sp.resize(psf, (psf_radius, psf_radius, 1)) for psf in target_psfs]

    # measured correct FWHM
    measured_fwhms = []
    for target_psf in target_psfs:
        measured_fwhm = get_FWHM_from_pixel(target_psf)
        measured_fwhms += [measured_fwhm[0]]
    # print('Measured FWHM', measured_fwhms)
    # quit()
    
    # add noise
    if args.noise != 0:
        np.random.seed(0)
        for i in range(len(target_images)):
            noise = np.random.normal(size=target_images[i].shape, scale=args.noise)
            target_images[i] += noise

    # get mask
    implant_mask = get_implant_mask(reference_image)
    mask = get_signal_mask(implant_mask)

    # map resolution
    psf_maps = []
    fwhm_maps = []
    stride = config['params']['psf-stride']
    num_workers = config['params']['num-workers']
    for i in range(len(target_images)):
        print('Mapping resolution for case {} of {} with sigma {}'.format(i+1, len(target_images), sigmas[i]))
        psf_map, fwhm_map = map_resolution(reference_image, target_images[i], patch_shape, resolution_mm, mask, stride, num_workers=num_workers)
        psf_maps.append(psf_map)
        fwhm_maps.append(fwhm_map) 
    psf_maps = np.stack(psf_maps)
    fwhm_maps = np.stack(fwhm_maps)
    
    volumes = [reference_image] + target_images
    fig1, tracker1 = plotVolumes(volumes)
    fig2, tracker2 = plotVolumes(target_psfs)

    # display results
    fig, tracker = plot_fwhm_maps(fwhm_maps / resolution_mm[0])
    plot_distribution(fwhm_maps / resolution_mm[0], measured_fwhms)

    # show an example patch for each case
    # TODO for each case, show one target & blurred patch, and the correct and estimated PSF

    plt.show()
