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
import psf_new
from util import safe_divide


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
        k_full = sp.fft(images[0].data)
        fullShape = k_full.shape
        for i in range(1, len(images)):
            acqShape = images[i].meta.acqMatrixShape
            print('hacking image {} to be copy of reference image {} with k-space shape {}'.format(i, fullShape, acqShape))
            noise = np.random.normal(size=acqShape, scale=8e2)
            k = sp.resize(sp.resize(k_full, acqShape) + noise, fullShape)
            # k = sp.resize(sp.resize(k_full, acqShape), fullShape)
            # k = sp.resize(sp.fft(images[i].data), fullShape)  # or just zero-pad original data to match reference
            images[i].data = np.abs(sp.ifft(k))

        # hack: interpolate images up to 512 x 512 resolution
        # shape = (512, 512, 64)
        # unit_cell_pixels = 2 * unit_cell_pixels
        # for i in range(1, len(images)):
        #     images[i].data = np.abs(sp.ifft(sp.resize(sp.fft(images[i].data), shape)))

        images = np.stack([image.data for image in images])

        # rescale data for comparison
        images[0] = analysis.normalize(images[0])
        for i in range(1, len(images)):
            images[i] = analysis.equalize(images[i], images[0])
        
        # plot line pairs
        line_pairs = False
        if line_pairs:
            slc_x = (slice(182, 214), slice(113, 145), 15)
            slc_y = (slice(182, 214), slice(151, 183), 15)
            fig, axes = plt.subplots(nrows=2, ncols=len(images)-1, figsize=(12, 5))
            # if len(images) == 2:
            #     axes_x = np.array([axes_x])
            #     axes_y = np.array([axes_y])
            plot_kwargs = {'vmin': 0, 'vmax': 1, 'cmap': 'gray'}
            for i in range(len(images)-1):
                j = i + 1
                print(images.shape)
                print(images[j].shape)
                print(slc_x)
                print(images[j][slc_x].shape)
                axes[0, i].imshow(images[j][slc_x], **plot_kwargs)
                axes[0, i].set_title('{} x {}\n'.format(shapes[j][0], shapes[j][1]), fontsize=16)
                axes[0, i].set_xlabel('{:.1f}'.format(shapes[0][0] / shapes[1+i][0]), fontsize=20)
                axes[1, i].imshow(images[j][slc_y], **plot_kwargs)
                axes[1, i].set_xlabel('{:.1f}'.format(shapes[0][1] / shapes[1+i][1]), fontsize=20)
                axes[0, i].set_xticks([])
                axes[0, i].set_yticks([])
                axes[1, i].set_xticks([])
                axes[1, i].set_yticks([])
            axes[0, 0].set_ylabel('X Line Pairs', fontsize=20)
            axes[1, 0].set_ylabel('Y Line Pairs', fontsize=20)
            plt.tight_layout()
            fig.savefig(path.join(save_dir, 'line_pairs.png'), dpi=300)

        # compute masks
        load_mask = True
        if load_mask:
            mask_psf = np.load(path.join(save_dir, 'mask_psf.npy'))
            print('mask shape', mask_psf.shape)
        else:
            if args.verbose:
                print('Computing masks...')
            mask_empty = analysis.get_mask_empty(images[0])
            mask_implant = analysis.get_mask_implant(mask_empty)

            metal = False
            if metal:
                mask_signal = analysis.get_mask_signal(images[0])
                signal_ref = analysis.get_typical_level(images[0], mask_signal, mask_implant)
                error = images[1] - images[0]
                normalized_error = safe_divide(error, signal_ref)
                mask_artifact = analysis.get_mask_artifact(normalized_error)
                mask_psf = analysis.get_mask_register(mask_empty, mask_implant, mask_artifact)
            else:
                mask_lattice = analysis.get_mask_lattice(images[0])
                mask_psf = np.logical_and(mask_lattice, ~mask_implant)
            np.save(path.join(save_dir, 'mask_psf.npy'), mask_psf)

        slc = (slice(35, 155), slice(65, 185), slice(15, 45))
        slc = (slice(35, 95), slice(65, 125), slice(20, 40))
        # slc = (slice(35, 65), slice(65, 95), slice(20, 40))
        # slc = (slice(35*2, 65*2), slice(65*2, 95*2), slice(20, 40))
        slc = (slice(35*2, 95*2), slice(65*2, 125*2), slice(20, 40))
        # slc = (slice(35*2, 155*2), slice(65*2, 185*2), slice(15, 45))
        # slc = (slice(35*2, 35*2+100), slice(65*2, 65*2+100), slice(5, 55))

        mask_psf = mask_psf[slc]
        images = images[(slice(None),) + slc]

        # fig, tracker = plotVolumes((images[0], images[1], images[2], images[3], mask_psf,))
        # plt.show()

        # compute PSF & FWHM
        num_trials = len(images) - 1
        patch_shape = (unit_cell_pixels,) * 3  # might want to double for better noise robustness
        psfs = []
        fwhms = []
        for i in range(num_trials):
            if args.verbose:
                print('on trial {} with acquired matrix shapes {} and {} Hz'.format(i, shapes[0], shapes[1+i]))
            # psf_i = psf.map_psf(images[0], images[1+i], mask_psf, patch_shape, None, args.stride, 'kspace', num_workers=8)
            psf_i = psf_new.estimate_psf(images[0], images[1+i], mask_psf, patch_shape, args.stride, 8)  # TODO PSF shape is hard-coded rn
            fwhm_i = fwh.get_FWHM_in_parallel(psf_i)
            for j in range(3):
                continue
                fwhm_i[..., j] = ndi.median_filter(fwhm_i[..., j], size=int(unit_cell_pixels * 0.3), mode='reflect')
                mask = (fwhm_i[..., j] > 0)
                # mask = ndi.generic_filter(mask, np.max, size=int(unit_cell_pixels * 0.4), mode='constant', cval=1) 
                mask = ndi.generic_filter(mask, np.min, size=int(unit_cell_pixels * 0.3), mode='constant', cval=1)  # erosion to remove edge effects from use of mean filter above (using generic instead of canonical binary_erosion for control over boundary conditions)
                fwhm_i[~mask, j] = 0
            psfs.append(psf_i)
            fwhms.append(fwhm_i)

        # save outputs
        if args.verbose:
            print('Saving outputs...')
        np.savez(path.join(save_dir, 'outputs.npz'),
            images=images,
            # erosion_mask=erosion_mask,
            shapes=shapes,
            psfs=psfs,
            fwhms=np.stack(fwhms),
            unit_cell_pixels=unit_cell_pixels
         )
    
    else:

        with open(path.join(save_dir, 'args_post.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=4)

        data = np.load(path.join(save_dir, 'outputs.npz'))
        for var in data:
            globals()[var] = data[var]

    num_trials = len(images) - 1

    fs = 18
    # matrix_shapes = ['256x256', '256x172', '256x128', '172x256', '172x172', '172x128', '128x256', '128x172', '128x128'] # TODO automate this
    matrix_shapes = ['{}x{}'.format(shape[0], shape[1]) for shape in shapes[1:]]

    fwhm_x_masked_list = [fwhms[i][..., 0][fwhms[i][..., 0] > 0] for i in range(num_trials)]
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 12))
    ax = axes[0]
    # sns.violinplot(fwhm_x_masked_list, ax=ax)
    sns.boxplot(fwhm_x_masked_list, ax=ax)
    ax.set_xlim([-0.5, 8.5])
    # ax.set_ylim([0.9, 3])
    # ax.set_yticks([1, 1.7, 2.4])
    ax.set_ylim([0.9, 6])
    ax.set_yticks([1, 1.7, 2.4, 3.6, 4.8])
    ax.set_xticks(range(len(matrix_shapes)))
    # ax.set_xticklabels(['1'] * 3 + ['1.5'] * 3 + ['2'] * 3)
    ax.set_xticklabels(['{:.1f}'.format(shapes[0][0] / shape[0]) for shape in shapes[1:]])
    ax.set_xlabel('Relative Voxel Size in X (voxels)', fontsize=fs)
    ax.set_ylabel('Measured FWHM (voxels)', fontsize=fs)
    ax.tick_params(labelsize=fs)
    ax.grid(axis='y')

    ax = axes[1]
    fwhm_y_masked_list = [fwhms[i][..., 1][fwhms[i][..., 1] > 0] for i in range(num_trials)]
    # ax = sns.violinplot(fwhm_y_masked_list, ax=ax)
    sns.boxplot(fwhm_y_masked_list, ax=ax)
    ax.set_xlim([-0.5, 8.5])
    # ax.set_ylim([0.9, 3])
    # ax.set_yticks([1, 1.68, 2.43])
    ax.set_ylim([0.9, 6])
    ax.set_yticks([1, 1.7, 2.4, 3.6, 4.8])
    ax.set_xticks(range(len(matrix_shapes)))
    ax.set_xticklabels(['{:.1f}'.format(shapes[0][1] / shape[1]) for shape in shapes[1:]])
    ax.set_xlabel('Relative Voxel Size in Y (voxels)', fontsize=fs)
    ax.set_ylabel('Measured FWHM (voxels)', fontsize=fs)
    ax.tick_params(labelsize=fs)
    ax.grid(axis='y')

    newax = ax.twiny()
    newax.set_frame_on(True)
    newax.patch.set_visible(False)
    newax.xaxis.set_ticks_position('bottom')
    newax.xaxis.set_label_position('bottom')
    newax.spines['bottom'].set_position(('outward', 70))
    newax.set_xlim([-0.5, 8.5])
    newax.set_xticks(range(len(matrix_shapes)))
    newax.set_xticklabels(matrix_shapes, rotation=30, ha='right', fontsize=fs*0.75)
    newax.set_xlabel('Matrix Shape   ', fontsize=fs)

    plt.subplots_adjust(hspace=0.3, top=0.95, bottom=0.2)
    plt.savefig(path.join(save_dir, 'resolution.png'), dpi=300)

    for i in range(num_trials):

        mask = (fwhms[i][..., 0] > 0)
        fwhm_median = tuple(np.median(fwhms[i][..., j][mask]) for j in range(3))
        fwhm_std = tuple(np.std(fwhms[i][..., j][mask]) for j in range(3))
        print('--------------------------')
        print('FWHM median +/- std')
        for m, s in zip(fwhm_median, fwhm_std):
            print('{:.2f} +/- {:.2f}'.format(m, s))
        print('--------------------------')

        # shape = psfs[i].shape
        # # slc = (int(shape[0] * 0.25), int(shape[1] * 0.25), int(shape[2] * 0.25))
        # idx = np.argmin(fwhms[i][..., 0])
        # slc = np.unravel_index(idx, fwhms[i].shape[:3])
        # print('fwhm min at {} with value {}'.format(slc, fwhms[i][..., 0][slc]))
        # print('psf', psfs[i].shape)
        # print('slc', slc)
        # print('psf sliced', psfs[i][35, 49, 13, ...].shape)
        # # psf_slc = np.abs(psfs[i][35, 49, 13, ...])
        # slc = (35, 49, 13)
        # psfi = psfs[i][slc]
        # fwhm_x, _, _ = fwh.get_FWHM_from_psf_3D(psfi / np.max(psfi))
        # print('psfi stats', np.median(np.abs(psfi)), np.max(psfi))
        # print('fwhm_x', fwhm_x)
        # # slc = (30, 9, 9)
        # psf_slc = np.abs(psfs[i][slc])
        # psf_slc = psf_slc / np.max(psf_slc)

        # volumes = (psf_slc, psf_slc)
        # titles = ('PSF with FWHM {} pixels'.format(fwhms[i][slc]), 'Same')
        # fig1, tracker1 = plotVolumes(volumes, titles=titles, figsize=(16, 8))

        # # slc = (slice(78, 88, None), slice(90, 100, None), slice(2, 12, None))
        # slc = (slice(70, 80, None), slice(98, 108, None), slice(13, 23, None))
        # im0 = images[0][slc]
        # im1 = images[1+i][slc]
        # k0 = np.abs(sp.fft(im0))
        # k0 = k0 / np.max(k0)
        # k1 = np.abs(sp.fft(im1))
        # k1 = k1 / np.max(k1)
        # q = np.abs(safe_divide(k1, k0, thresh=1e-2))
        # q = q / np.max(q)
        # p = psf.estimate_psf_kspace(im0, im1)
        # qq = np.real(sp.ifft(q))
        # qq = qq / np.max(np.abs(qq))
        # volumes = (im0, im1, p / np.max(np.abs(p)), psfi / np.max(np.abs(psfi)), k0, k1, q, qq)
        # titles = ('Input image', 'Output image', 'PSF now', 'PSF got')
        # fig2, tracker2 = plotVolumes(volumes, titles=titles, figsize=(16, 8))

        volumes = (fwhms[i][..., 0], fwhms[i][..., 1], fwhms[i][..., 2])
        titles = ('FWHM in x', 'FWHM in y', 'FWHM in z')
        # fig3, tracker3 = plotVolumes(volumes, titles=titles, figsize=(16, 8), vmin=0, vmax=10, cmap='tab20c', cbar=True)
        fig3, tracker3 = plotVolumes(volumes, titles=titles, figsize=(16, 5), vmin=0, vmax=6, cmap='viridis', cbar=True)
    
    plt.show()