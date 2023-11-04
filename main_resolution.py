import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
from os import path, makedirs
from pathlib import Path
import seaborn as sns
import sigpy as sp
from skimage import morphology
from time import time

import analysis
import dicom
import fwhm as fwh
from plot import plotVolumes
import psf


p = argparse.ArgumentParser(description='Resolution analysis of image volumes with common dimensions.')
p.add_argument('root', type=str, help='path where outputs are saved')
p.add_argument('-e', '--exam_root', type=str, default=None, help='directory where exam data exists in subdirectories')
p.add_argument('-s', '--series_list', type=str, nargs='+', default=None, help='list of exam_root subdirectories to be analyzed, with the first serving as reference')
p.add_argument('-c', '--unit_cell_mm', type=float, default=12.0, help='size of lattice unit cell (in mm)')
p.add_argument('-t', '--stride', type=int, default=2, help='window stride length for stepping between PSF measurements')
p.add_argument('-v', '--verbose', action='count', default=0, help='verbosity level')

def load_dicom_series(path):
    if path is None:
        return None
    files = Path(path).glob('*MRDC*')
    image = dicom.load_series(files)
    return image

if __name__ == '__main__':

    args = p.parse_args()
    
    # set up directory structure
    save_dir = path.join(args.root, 'resolution')
    if not path.exists(save_dir):
        makedirs(save_dir)

    if args.exam_root is not None and args.series_list is not None:

        with open(path.join(save_dir, 'args.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=4)

        # load data
        images = []
        for series_name in args.series_list:
            image = load_dicom_series(path.join(args.exam_root, series_name))
            images.append(image)
            if args.verbose:
                print('Found DICOM series {}; loaded data with shape {}'.format(series_name, image.shape))

        # extract relevant metadata and throw away the rest
        shapes = np.stack([np.array(image.meta.acqMatrixShape) for image in images])
        if images[0].is_isotropic:
            voxel_size_mm = images[0].meta.resolution_mm[0]
        else:
            raise ValueError('Isotropic resolution is required, but got: ', images[0].meta.resolution_mm)
        unit_cell_pixels = int(args.unit_cell_mm / voxel_size_mm)
        print('From voxel size {} mm, compute unit cell size is {} pixels'.format(voxel_size_mm, unit_cell_pixels))

        # hack: overwrite images to be copies of the reference with masked k-space 
        # k_full = sp.fft(images[0].data)
        # fullShape = k_full.shape
        # for i in range(1, len(images)):
        #     acqShape = images[i].meta.acqMatrixShape
        #     print('hacking image {} to be copy of reference image {} with k-space shape {}'.format(i, fullShape, acqShape))
        #     k = sp.resize(sp.resize(k_full, acqShape), fullShape)
        #     images[i].data = np.abs(sp.ifft(k))

        images = np.stack([image.data for image in images])

        # rescale data for comparison
        images[0] = analysis.normalize(images[0])
        for i in range(1, len(images)):
            images[i] = analysis.equalize(images[i], images[0])
        
        # plot line pairs
        slc_x = (slice(182, 214), slice(113, 145), 15)
        slc_y = (slice(182, 214), slice(151, 183), 15)
        fig_x, axes_x = plt.subplots(nrows=1, ncols=len(images)-1, figsize=(12, 3))
        fig_y, axes_y = plt.subplots(nrows=1, ncols=len(images)-1, figsize=(12, 3))
        if len(images) == 2:
            axes_x = np.array([axes_x])
            axes_y = np.array([axes_y])
        plot_kwargs = {'vmin': 0, 'vmax': 1, 'cmap': 'gray'}
        for i in range(len(images)-1):
            j = i + 1
            print(images.shape)
            print(images[j].shape)
            print(slc_x)
            print(images[j][slc_x].shape)
            axes_x[i].imshow(images[j][slc_x], **plot_kwargs)
            axes_x[i].set_title('{} x {}'.format(shapes[j][0], shapes[j][1]))
            axes_y[i].imshow(images[j][slc_y], **plot_kwargs)
            axes_y[i].set_title('{} x {}'.format(shapes[j][0], shapes[j][1]))
            axes_x[i].set_xticks([])
            axes_x[i].set_yticks([])
            axes_y[i].set_xticks([])
            axes_y[i].set_yticks([])
        fig_x.savefig(path.join(save_dir, 'line_pairs_x.png'))
        fig_y.savefig(path.join(save_dir, 'line_pairs_y.png'))

        # compute masks
        if args.verbose:
            print('Computing masks...')
        mask_empty = analysis.get_mask_empty(images[0])
        mask_implant = analysis.get_mask_implant(mask_empty)
        mask_lattice = analysis.get_mask_lattice(images[0])
        mask_psf = np.logical_and(mask_lattice, ~mask_implant)

        slc = (slice(35, 155), slice(65, 185), slice(15, 45))

        mask_psf = mask_psf[slc]
        images = images[(slice(None),) + slc]

        # compute PSF & FWHM
        num_trials = len(images) - 1
        patch_shape = (unit_cell_pixels,) * 3
        psfs = []
        fwhms = []
        for i in range(num_trials):
            if args.verbose:
                print('on trial {} with acquired matrix shapes {} and {} Hz'.format(i, shapes[0], shapes[1+i]))
            psf_i = psf.map_psf(images[0], images[1+i], mask_psf, patch_shape, None, args.stride, 'kspace', num_workers=8)
            fwhm_i = fwh.get_FWHM_in_parallel(psf_i)
            for j in range(3):
                fwhm_i[..., j] = ndi.median_filter(fwhm_i[..., j], size=int(unit_cell_pixels))
            psfs.append(psf_i)
            fwhms.append(fwhm_i)

        # save outputs
        if args.verbose:
            print('Saving outputs...')
        np.savez(path.join(save_dir, 'outputs.npz'),
            images=images,
            mask_psf=mask_psf,
            shapes=shapes,
            psfs=psfs,
            fwhms=np.stack(fwhms),
            voxel_size_mm=voxel_size_mm,
            unit_cell_pixels=unit_cell_pixels
         )
    
    else:

        with open(path.join(save_dir, 'args_post.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=4)

        data = np.load(path.join(save_dir, 'outputs.npz'))
        for var in data:
            globals()[var] = data[var]

    num_trials = len(images) - 1

    mask_psf_eroded = ndi.binary_erosion(fwhms[0][..., 0] > 0, structure=morphology.cube(int(unit_cell_pixels)))
    for i in range(num_trials):
        for j in range(3):
            fwhms[i][~mask_psf_eroded, j] = 0

    fwhm_x_masked_list = [fwhms[i][..., 0][fwhms[i][..., 0] > 0] for i in range(num_trials)]
    plt.figure()
    sns.violinplot(fwhm_x_masked_list)
    plt.savefig(path.join(save_dir, 'resolution_x.png'))

    fwhm_y_masked_list = [fwhms[i][..., 1][fwhms[i][..., 1] > 0] for i in range(num_trials)]
    plt.figure()
    sns.violinplot(fwhm_y_masked_list)
    plt.savefig(path.join(save_dir, 'resolution_y.png'))

    for i in range(num_trials):

        mask = (fwhms[i][..., 0] > 0)
        fwhm_median = tuple(np.median(fwhms[i][..., j][mask]) for j in range(3))
        fwhm_std = tuple(np.std(fwhms[i][..., j][mask]) for j in range(3))
        print('--------------------------')
        print('FWHM median +/- std')
        for m, s in zip(fwhm_median, fwhm_std):
            print('{:.2f} +/- {:.2f}'.format(m, s))
        print('--------------------------')

        shape = psfs[i].shape
        slc = (int(shape[0] * 0.25), int(shape[1] * 0.25), int(shape[2] * 0.25))
        psf_slc = np.abs(psfs[i][slc])
        psf_slc = psf_slc / np.max(psf_slc)

        volumes = (psf_slc, psf_slc)
        titles = ('PSF with FWHM {} pixels'.format(fwhms[i][slc]), 'Same')
        fig1, tracker1 = plotVolumes(volumes, titles=titles, figsize=(16, 8))

        volumes = (images[0], images[1+i], mask_psf)
        titles = ('Input image', 'Output image', 'PSF mask')
        fig2, tracker2 = plotVolumes(volumes, titles=titles, figsize=(16, 8))

        volumes = (fwhms[i][..., 0], fwhms[i][..., 1], fwhms[i][..., 2])
        titles = ('FWHM in x', 'FWHM in y', 'FWHM in z')
        # fig3, tracker3 = plotVolumes(volumes, titles=titles, figsize=(16, 8), vmin=0, vmax=10, cmap='tab20c', cbar=True)
        fig3, tracker3 = plotVolumes(volumes, titles=titles, figsize=(16, 5), vmin=0, vmax=4, cmap='viridis', cbar=True)
    
    plt.show()