import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import sigpy as sp
from os import path, makedirs
from pathlib import Path

import analysis
import dicom
import psf
import resolution
from plot import plotVolumes
from plot_resolution import box_plots


p = argparse.ArgumentParser(description='Resolution analysis of image volumes with common dimensions.')
p.add_argument('root', type=str, help='path where outputs are saved')
p.add_argument('-e', '--exam_root', type=str, default=None, help='directory where exam data exists in subdirectories')
p.add_argument('-s', '--series_list', type=str, nargs='+', default=None, help='list of exam_root subdirectories to be analyzed, with the first serving as reference')
p.add_argument('-c', '--unit_cell_mm', type=float, default=12.0, help='size of lattice unit cell (in mm)')
p.add_argument('-t', '--stride', type=int, default=2, help='window stride length for stepping between PSF measurements')
p.add_argument('-n', '--noise', type=float, default=0, help='st. dev. of noise added to overwritten k-space; default = 0')
p.add_argument('-w', '--overwrite', action='store_true', help='overwrite target k-space with samples from reference')
p.add_argument('-m', '--mask', action='store_true', help='re-use mask if one exists')
p.add_argument('-v', '--verbose', action='count', default=0, help='verbosity level')

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
            files = Path(path.join(args.exam_root, series_name)).glob('*MRDC*')
            image = dicom.load_series(files)
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

        ref_shape = images[0].data.shape
        if args.overwrite:
            k_ref = sp.fft(images[0].data)
            for i in range(1, len(images)):
                acqShape = images[i].meta.acqMatrixShape
                k_acq = sp.resize(k_ref, acqShape)
                if args.noise != 0:
                    noise = np.random.normal(size=acqShape, scale=args.noise)
                    k_acq += noise
                images[i].data = np.abs(sp.ifft(sp.resize(k_acq, ref_shape)))
        else:
            for i in range(1, len(images)):
                k_acq = sp.fft(images[i].data)
                images[i].data = np.abs(sp.ifft(sp.resize(k_acq, ref_shape)))

        images = np.stack([image.data for image in images])

        # rescale data for comparison
        images[0] = analysis.normalize(images[0])
        for i in range(1, len(images)):
            images[i] = analysis.equalize(images[i], images[0])
        
        # compute masks
        mask_file = path.join(save_dir, 'mask.npy')
        if args.mask and path.isfile(mask_file):
            print('Loading pre-computed mask...')
            mask = np.load(mask_file)
        else:
            print('Computing mask...')
            mask = resolution.get_mask(images[0], images[1], metal=False)
            np.save(mask_file, mask)

        slc = (slice(35, 155), slice(65, 185), slice(15, 45))
        slc = (slice(35, 95), slice(65, 125), slice(20, 40))
        slc = (slice(35*2, 95*2), slice(65*2, 125*2), slice(20, 40))

        mask = mask[slc]
        images = images[(slice(None),) + slc]

        # compute PSF & FWHM
        num_trials = len(images) - 1
        psfs = []
        fwhms = []
        for i in range(num_trials):
            if args.verbose:
                print('on trial {} with acquired matrix shapes {} and {} Hz'.format(i, shapes[0], shapes[1+i]))
            psf_i, fwhm_i = psf.map_resolution(
                images[0],
                images[1+i],
                unit_cell_pixels,
                stride=args.stride,
                num_workers=8,
                mask=mask
                )
            psfs.append(psf_i)
            fwhms.append(fwhm_i)
        
        psfs = np.stack(fwhms)
        fwhms = np.stack(fwhms)

        # save outputs
        if args.verbose:
            print('Saving outputs...')
        np.savez(path.join(save_dir, 'outputs.npz'),
            images=images,
            shapes=shapes,
            psfs=psfs,
            fwhms=fwhms,
            unit_cell_pixels=unit_cell_pixels
         )
    
    else:

        with open(path.join(save_dir, 'args_post.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=4)

        data = np.load(path.join(save_dir, 'outputs.npz'))
        for var in data:
            globals()[var] = data[var]
    
    box_plots(fwhms, shapes, save_dir=save_dir)

    for i in range(num_trials):
        volumes = (fwhms[i][..., 0], fwhms[i][..., 1], fwhms[i][..., 2])
        titles = ('FWHM in x', 'FWHM in y', 'FWHM in z')
        fig, tracker = plotVolumes(volumes, titles=titles, figsize=(12, 4), vmin=0, vmax=6, cmap='viridis', cbar=True)
    
    plt.show()