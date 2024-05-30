"""Make Figure 9 for paper.

"""
import argparse
import matplotlib.pyplot as plt
import numpy as np
import sigpy as sp
import yaml

from os import path, makedirs
from matplotlib.ticker import MultipleLocator

from toniq import snr, sr
from toniq.config import parse_slice, load_volume
from toniq.masks import get_implant_mask, get_signal_mask
from toniq.plot import color_panels, label_panels, remove_ticks, label_encode_dirs, plotVolumes
from toniq.plot_params import *
from toniq.util import normalize, safe_divide

def plot_fwhm_maps(maps):
    volumes = []
    for i in range(2):
        for j in range(len(maps)):
            volumes += [maps[j][..., i]]
    fig, tracker = plotVolumes(volumes, figsize=(12, 4), nrows=2, ncols=len(maps), vmin=1, vmax=3, cmap=CMAP['resolution'], cbar=True)
    return fig, tracker

def print_statistics(fwhm_maps, targets):
    for i in range(len(fwhm_maps)):
        distr = fwhm_maps[i]
        distr = distr[distr > 0]
        print('Trial {}: mean bias {:.2f}, std {:.2f}, max error {:.2f}'.format(
            i, np.mean(distr)-targets[i], np.std(distr), np.max(np.abs(distr-targets[i]))))

def plot_distribution(ax, targets, maps, major_grid=True, minor_grid=True):
    maps = [maps[i][maps[i]>0] for i in range(len(maps))]
    ax.violinplot(maps, showmedians=True)
    ax.set_xlabel('Simulation Trials')
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
            image = safe_divide(np.abs(image), np.max(np.abs(image)))
        im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        ims += [im]
    return ims

p = argparse.ArgumentParser(description='Make Figure 9')
p.add_argument('save_dir', type=str, help='path where outputs are saved')
p.add_argument('-c', '--config', type=str, default='fse125.yml', help='data config file')
p.add_argument('-n', '--noise', type=float, default=0.01, help='standard deviation of noise added to normalized image; default=0.01')
p.add_argument('-l', '--load', action='store_true', help='load inputs from save_dir')
p.add_argument('-p', '--plot', action='store_true', help='show plots')
p.add_argument('-x', type=int, default=105, help='left/right coordinate of inset location; default=105')
p.add_argument('-y', type=int, default=88, help='up/down coordinate of inset location; default=88')
p.add_argument('-z', '--z_slice', type=int, default=18, help='relative position of FWHM z slice (after crop); default=18')
p.add_argument('--psf_window_size', type=int, nargs=3, default=[14, 14, 10], help='size of window used for SR mapping; default=[14, 14, 10]')
p.add_argument('--psf_shape', type=int, nargs=3, default=[5, 5, 1], help='size of PSF used for SR mapping; default=[5, 5, 1]')
p.add_argument('--psf_stride', type=int, default=1, help='stride used for SR mapping; default=1')
p.add_argument('--num_workers', type=int, default=8, help='number of workers used for SR mapping; default=8')
p.add_argument('--blur_radius', type=int, default=20, help='Radius of blurring PSF; default=20')
p.add_argument('-s', '--sigma', type=float, nargs='+', default=[0.3, 0.558, 0.675, 0.769, 0.85, 0.926, 1.02, 1.128, 1.247], help='st. dev. for blurring PSFs (gaussian); default = sigmas yielding FWHM=1:3:0.25 pixels')


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
        reference_image = load_volume(config, 'structured-plastic-reference').data
        # resolution_mm = image.meta.resolution_mm
        resolution_mm = [1.2, 1.2, 1.2]
        reference_image = np.abs(sp.ifft(sp.resize(sp.fft(reference_image), (256, 256, 64))))
        reference_image = normalize(reference_image)

        # generate blurred targets and associated PSFs
        target_images = []
        target_psfs = []
        for sigma in args.sigma:
            target_images += [sr.gaussian_blur(reference_image, sigma, args.blur_radius)]
            target_psfs += [sr.gaussian_psf(reference_image.shape, sigma, args.blur_radius)]

        # reduce image to target slice of interest
        reference_image = reference_image[slc]
        target_images = [image[slc] for image in target_images]
        target_psfs = [sp.resize(psf, (args.blur_radius, args.blur_radius, 1)) for psf in target_psfs]

        # measured correct FWHM
        measured_fwhms = []
        for target_psf in target_psfs:
            measured_fwhm = sr.measure_fwhm(target_psf)
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
            snr_i = snr.get_map(target_images[i], target_images_2[i], mask)
            snr_maps += [snr_i]
            print('Trial {} has SNR stats: (min {:.2f}, mean {:.2f}, median {:.2f}, max {:.2f})'.format(i, np.min(snr_i), np.mean(snr_i), np.median(snr_i), np.max(snr_i)))

        # map resolution
        psf_maps = []
        fwhm_maps = []
        for i in range(len(target_images)):
            print('Mapping resolution for case {} of {} with sigma {}'.format(i+1, len(target_images), args.sigma[i]))
            psf_map, fwhm_map = sr.get_map(reference_image, target_images[i], tuple(args.psf_shape), tuple(args.psf_window_size), resolution_mm, mask, args.psf_stride, num_workers=args.num_workers)
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
    # fig, tracker = plotVolumes((fwhm_maps[-1][..., 0] / resolution_mm[0],), vmin=2.875, vmax=3.125)

    fig = plt.figure(figsize=(FIG_WIDTH[2], FIG_WIDTH[2] * 0.8))
    subfigs = fig.subfigures(2, 1, hspace=0.03, height_ratios=[1.8, 1])
    subfigs[0].suptitle('Simulation Trials')

    axes = subfigs[0].subplots(nrows=4, ncols=len(args.sigma), gridspec_kw={'wspace': 0, 'hspace': 0, 'bottom': 0.05, 'right': 0.94})
    inset = (slice(args.y, args.y + args.psf_window_size[0] - args.psf_shape[0] + 1),
             slice(args.x, args.x + args.psf_window_size[1] - args.psf_shape[1] + 1),
             args.z_slice)
    plot_row(axes[0, :], target_psfs, shape=args.psf_shape[:2], normalize=True)
    plot_row(axes[1, :], target_images, slc=inset)
    plot_row(axes[2, :], psf_maps, slc=(inset[0].start, inset[1].start, inset[2]), normalize=True)
    ims = plot_row(axes[3, :], fwhm_maps / resolution_mm[0], slc=(slice(None), slice(None), inset[2], 0), vmin=0.75, vmax=3.25, cmap=CMAP['resolution'])
    for ax, label in zip(axes[:, 0], ('Simulated\nPSF', 'Target\nPatch', 'Local PSF\nEstimate', 'x\nResolution\nMap')):
        ax.set_ylabel(label, rotation='horizontal', va='center', ha='center', labelpad=30)
    for ax, title in zip(axes[0, 1:], ['{:.2f}'.format(i) for i in np.arange(1.25, 3.1, 0.25)]):
        ax.set_title(title)
    label_encode_dirs(axes[0, 0], color='white', offset=0.01)
    axes[0, 0].set_title('FWHM =\n1.00')
    remove_ticks(axes)
    sr.colorbar(axes[3, -1], ims[-1], 'FWHM (pixels)')

    axes = subfigs[1].subplots(nrows=1, ncols=2, gridspec_kw={'bottom': 0.15, 'top': 0.85, 'right': 0.94})
    plot_distribution(axes[0], measured_fwhms, [maps[..., 0] / resolution_mm[0] for maps in fwhm_maps])
    plot_distribution(axes[1], (1.00,), [maps[..., 1] / resolution_mm[1] for maps in fwhm_maps])
    axes[0].set_title('x Resolution')
    axes[1].set_title('y Resolution')

    # print_statistics(fwhm_maps[..., 0] / resolution_mm[0], np.arange(1, 3.1, 0.25))
    # print_statistics(fwhm_maps[..., 1] / resolution_mm[1], np.ones(9))

    color_panels(subfigs.flat)
    label_panels(subfigs.flat)

    plt.savefig(path.join(args.save_dir, 'figure9.png'), dpi=DPI)
    plt.savefig(path.join(args.save_dir, 'figure9.pdf'), dpi=DPI)

    if args.plot:
        plt.show()
