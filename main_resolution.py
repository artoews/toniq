import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
from os import path, makedirs
from pathlib import Path
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
        images = np.stack([image.data for image in images])

        # rescale data for comparison
        images[0] = analysis.normalize(images[0])
        for i in range(1, len(images)):
            images[i] = analysis.equalize(images[i], images[0])

        # compute masks
        if args.verbose:
            print('Computing masks...')
        mask_lattice = analysis.get_mask_lattice(images[0])

        # TODO is there any need for cropping if we are using the mask_lattice anyway?

        # compute PSF & FWHM
        num_trials = len(images) - 1
        patch_shape = (unit_cell_pixels,) * 3
        psfs = []
        fwhms = []
        for i in range(num_trials):
            if args.verbose:
                print('on trial {} with acquired matrix shapes {} and {} Hz'.format(i, shapes[0], shapes[1+i]))
            psf_i = psf.map_psf(images[0], images[1+i], mask_lattice, patch_shape, None, args.stride, 'kspace', num_workers=8)
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
            mask_lattice=mask_lattice,
            shapes=shapes,
            psfs=psfs,
            fwhms=fwhms,
            voxel_size_mm=voxel_size_mm
         )
    
    else:

        with open(path.join(save_dir, 'args_post.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=4)

        data = np.load(path.join(save_dir, 'outputs.npz'))
        for var in data:
            globals()[var] = data[var]

    num_trials = len(images) - 1

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
        slc = (int(shape[0] * 0.5), int(shape[1] * 0.5), int(shape[2] * 0.5))
        psf_slc = np.abs(psfs[i][slc])
        psf_slc = psf_slc / np.max(psf_slc)

        volumes = (psf_slc, psf_slc)
        titles = ('PSF with FWHM {} pixels'.format(fwhms[i][slc]), 'Same')
        fig1, tracker1 = plotVolumes(volumes, titles=titles, figsize=(16, 8))

        volumes = (images[0], images[i], mask_lattice)
        titles = ('Input image', 'Output image', 'Lattice mask')
        fig2, tracker2 = plotVolumes(volumes, titles=titles, figsize=(16, 8))

        volumes = (fwhms[i][..., 0], fwhms[i][..., 1], fwhms[i][..., 2])
        titles = ('FWHM in x', 'FWHM in y', 'FWHM in z')
        # fig3, tracker3 = plotVolumes(volumes, titles=titles, figsize=(16, 8), vmin=0, vmax=10, cmap='tab20c', cbar=True)
        fig3, tracker3 = plotVolumes(volumes, titles=titles, figsize=(16, 5), vmin=0, vmax=4, cmap='viridis', cbar=True)
    
    plt.show()