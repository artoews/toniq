import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import sigpy as sp
from os import path, makedirs

from resolution import map_resolution, get_resolution_mask
from plot import plotVolumes
from plot_resolution import box_plots
from util import equalize, load_series

# TODO systematize this
# slc = (slice(35, 155), slice(65, 185), slice(15, 45))
# slc = (slice(40, 160), slice(65, 185), slice(15, 45))
# slc = (slice(35, 95), slice(65, 125), slice(20, 40))
# slc = (slice(35*2, 95*2), slice(65*2, 125*2), slice(20, 40))
slc = (slice(40*2, 160*2), slice(65*2, 185*2), slice(15, 45))

p = argparse.ArgumentParser(description='Resolution analysis of image volumes with common dimensions.')
p.add_argument('root', type=str, help='path where outputs are saved')
p.add_argument('-e', '--exam_root', type=str, default=None, help='directory where exam data exists in subdirectories')
p.add_argument('-s', '--series_list', type=str, nargs='+', default=None, help='list of exam_root subdirectories to be analyzed, with the first serving as reference')
p.add_argument('-c', '--unit_cell_mm', type=float, default=12.0, help='size of lattice unit cell (in mm); default=12')
p.add_argument('-t', '--stride', type=float, default=1, help='window stride length for stepping between PSF measurements, in units of unit cell length; default=1')
p.add_argument('-n', '--noise', type=float, default=0, help='st. dev. of noise added to k-space; default=0')
p.add_argument('-o', '--overwrite', action='store_true', help='overwrite target k-space with samples from reference')
p.add_argument('-w', '--workers', type=int, default=8, help='number of parallel pool workers; default=8')
p.add_argument('-m', '--mask', action='store_true', help='re-use mask if one exists')

if __name__ == '__main__':

    args = p.parse_args()

    save_dir = path.join(args.root, 'resolution')
    if not path.exists(save_dir):
        makedirs(save_dir)

    if args.exam_root is not None and args.series_list is not None:

        with open(path.join(save_dir, 'args.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=4)

        images = [load_series(args.exam_root, series_name) for series_name in args.series_list]

        matrix_shapes = np.stack([np.array(image.meta.acqMatrixShape) for image in images])

        resolution_mm = images[0].meta.resolution_mm
        unit_cell_pixels = np.array([int(args.unit_cell_mm / r) for r in resolution_mm])

        if args.overwrite:
            images = [images[0].data for _ in images]
        else:
            images = [image.data for image in images]

        for i in range(1, len(images)):
            k = sp.resize(sp.fft(images[i]), matrix_shapes[i])
            if args.noise != 0:
                k += np.random.normal(size=matrix_shapes[i], scale=args.noise)
            images[i] = np.abs(sp.ifft(sp.resize(k, matrix_shapes[0])))
        
        images = np.stack(images)

        images = equalize(images)

        mask_file = path.join(save_dir, 'mask.npy')
        if args.mask and path.isfile(mask_file):
            print('Loading pre-computed mask...')
            mask = np.load(mask_file)
        else:
            print('Computing mask...')
            mask = get_resolution_mask(images[0])
            np.save(mask_file, mask)
        
        if slc is not None:
            mask = mask[slc]
            images = images[(slice(None),) + slc]
        
        psfs = []
        fwhms = []
        for i in range(1, len(images)):
            print('working on matrix shape {} with reference {}'.format(i, matrix_shapes[i], matrix_shapes[0]))
            psf_i, fwhm_i = map_resolution(
                images[0],
                images[i],
                unit_cell_pixels,
                resolution_mm,
                stride=args.stride,
                num_workers=args.workers,
                mask=mask
                )
            psfs.append(psf_i)
            fwhms.append(fwhm_i) 
        psfs = np.stack(psfs)
        fwhms = np.stack(fwhms)

        np.savez(path.join(save_dir, 'outputs.npz'),
            images=images,
            matrix_shapes=matrix_shapes,
            psfs=psfs,
            fwhms=fwhms,
            resolution_mm=resolution_mm
         )
    
    else:

        with open(path.join(save_dir, 'args_post.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=4)

        data = np.load(path.join(save_dir, 'outputs.npz'))
        for var in data:
            globals()[var] = data[var]
    
    box_plots(fwhms / resolution_mm[0], matrix_shapes, save_dir=save_dir)

    # fig0, tracker0 = plotVolumes((images[0], images[1], images[2], images[3]))

    figs = [None] * len(fwhms)
    trackers = [None] * len(fwhms)
    for i in range(len(fwhms)):
        volumes = (fwhms[i][..., 0], fwhms[i][..., 1], fwhms[i][..., 2])
        titles = ('FWHM in x (mm)', 'FWHM in y (mm)', 'FWHM in z (mm)')
        figs[i], trackers[i] = plotVolumes(volumes, titles=titles, figsize=(12, 4), vmin=0.5, vmax=3.5, cmap='viridis', cbar=True)
    
    plt.show()