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
import resolution
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

    print(args.series_list)
    print(args.exam_root)
    print(save_dir)

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

        images = np.stack([image.data for image in images])

        # rescale data for comparison
        images[0] = analysis.normalize(images[0])
        for i in range(1, len(images)):
            images[i] = analysis.equalize(images[i], images[0])
        
        # compute masks
        load_mask = True
        if load_mask:
            mask = np.load(path.join(save_dir, 'mask_psf.npy'))
        else:
            metal = False
            mask = resolution.get_mask(images[0], images[1], metal=metal)
            np.save(path.join(save_dir, 'mask_psf.npy'), mask)

        slc = (slice(35, 155), slice(65, 185), slice(15, 45))
        slc = (slice(35, 95), slice(65, 125), slice(20, 40))
        # slc = (slice(35, 65), slice(65, 95), slice(20, 40))
        # slc = (slice(35*2, 65*2), slice(65*2, 95*2), slice(20, 40))
        slc = (slice(35*2, 95*2), slice(65*2, 125*2), slice(20, 40))
        # slc = (slice(35*2, 155*2), slice(65*2, 185*2), slice(15, 45))
        # slc = (slice(35*2, 35*2+100), slice(65*2, 65*2+100), slice(5, 55))

        mask = mask[slc]
        images = images[(slice(None),) + slc]

        # compute PSF & FWHM
        num_trials = len(images) - 1
        psfs = []
        fwhms = []
        for i in range(num_trials):
            if args.verbose:
                print('on trial {} with acquired matrix shapes {} and {} Hz'.format(i, shapes[0], shapes[1+i]))
            psf_i, fwhm_i = resolution.map_resolution(
                images[0],
                images[1+i],
                unit_cell_pixels,
                stride=args.stride,
                num_batches=8,
                mask=mask
                )
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

        volumes = (fwhms[i][..., 0], fwhms[i][..., 1], fwhms[i][..., 2])
        titles = ('FWHM in x', 'FWHM in y', 'FWHM in z')
        fig3, tracker3 = plotVolumes(volumes, titles=titles, figsize=(16, 5), vmin=0, vmax=6, cmap='viridis', cbar=True)
    
    plt.show()