import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from os import path, makedirs
import sigpy as sp
import scipy.ndimage as ndi
from skimage import morphology, transform

from masks import get_signal_mask, get_artifact_mask
from resolution import map_resolution, get_FWHM_from_pixel
from plot import plotVolumes
from plot_resolution import box_plots, plot_fwhm
from util import equalize, load_series, save_args
from slice_params import *
from plot_params import *

# slc = tuple(slice(s.start*2, s.stop*2) for s in LATTICE_SLC[:2]) + (LATTICE_SLC[2],)
# patch_shape = (20, 20, 10)
slc = LATTICE_SLC
patch_shape = (10, 10, 10)

p = argparse.ArgumentParser(description='Resolution analysis of image volumes with common dimensions.')
p.add_argument('root', type=str, help='path where outputs are saved')
p.add_argument('-e', '--exam_root', type=str, default=None, help='directory where exam data exists in subdirectories')
p.add_argument('-s', '--series_list', type=str, nargs='+', default=None, help='list of exam_root subdirectories to be analyzed, with the first serving as reference')
p.add_argument('--stride', type=int, default=5, help='window stride length for stepping between PSF measurements, in units of pixels; default=5')
p.add_argument('-n', '--noise', type=float, default=0, help='st. dev. of noise added to k-space; default=0')
p.add_argument('-o', '--overwrite', action='store_true', help='overwrite target k-space with samples from reference')
p.add_argument('-w', '--workers', type=int, default=8, help='number of parallel pool workers; default=8')
p.add_argument('-m', '--mask', action='store_true', help='re-use mask if one exists')
p.add_argument('-p', '--plot', action='store_true', help='show plots')
p.add_argument('-t', '--threshold', type=float, default=None, help='maximum intensity artifact error included in mask; default=None')

if __name__ == '__main__':

    args = p.parse_args()

    save_dir = path.join(args.root, 'resolution')
    artifact_dir = path.join(args.root, 'artifact')
    distortion_dir = path.join(args.root, 'distortion')
    if not path.exists(save_dir):
        makedirs(save_dir)

    if args.exam_root is not None and args.series_list is not None:

        save_args(args, save_dir)

        images = [load_series(args.exam_root, series_name) for series_name in args.series_list]
        matrix_shapes = np.stack([np.array(image.meta.acqMatrixShape) for image in images])
        matrix_shapes[1:, 0] = np.array([[240, 230, 220, 210, 200, 190, 180, 170]])
        # resolution_mm = images[0].meta.resolution_mm
        resolution_mm = images[1].meta.resolution_mm

        if args.overwrite:
            images = [images[0].data for _ in images]
            # images[0] = np.abs(sp.ifft(sp.resize(sp.resize(sp.fft(images[0]), matrix_shapes[1]), matrix_shapes[0]))) # TODO just debugging see if outer k-space does anything 
        else:
            images = [image.data for image in images]

        # TODO temporary to try downsampling high-res reference; TODO also compare with just using a 256x256 image here
        images[0] = np.abs(sp.ifft(sp.resize(sp.fft(images[0]), (256, 256, 64))))
        matrix_shapes[0] = images[0].shape

        print(matrix_shapes)

        for i in range(1, len(images)):
            k = sp.fft(images[i])
            if args.overwrite:
                k = sp.resize(k, matrix_shapes[i])
            images[i] = np.abs(sp.ifft(sp.resize(k, matrix_shapes[0])))
        
        images = np.stack(images)
        images = equalize(images)

        if args.noise != 0:
            for i in range(1, len(images)):
                k = sp.fft(images[i])
                # noise = np.random.normal(size=matrix_shapes[1], scale=args.noise) # assumes all non-reference images have same DICOM array shape, even if k-space was undersampled
                noise = np.random.normal(size=k.shape, scale=args.noise) # assumes all non-reference images have same DICOM array shape, even if k-space was undersampled
                k += sp.resize(noise, k.shape)
                images[i] = np.abs(sp.ifft(k))

        images = images[(slice(None),) + slc]

        implant_mask = np.load(path.join(artifact_dir, 'implant-mask.npy'))
        if args.threshold is None:
            mask = get_signal_mask(implant_mask)
        else:
            ia_maps = np.load(path.join(artifact_dir, 'ia-maps.npy'))
            gd_maps = np.load(path.join(distortion_dir, 'gd-maps.npy'))
            ia_mask = get_artifact_mask(ia_maps[0], args.threshold)
            gd_masks = [get_artifact_mask(gd_maps[0][..., i], 1) for i in range(3)]
            mask = get_signal_mask(implant_mask, artifact_masks=[ia_mask] + gd_masks)
        mask = transform.resize(mask, images[0].shape)

        # titles = ['{}x{}'.format(shape[0], shape[1]) for shape in matrix_shapes]
        # fig0, tracker0 = plotVolumes((images[0], images[1]), titles=titles[:2])
        # fig1, tracker1 = plotVolumes((images[0], mask))
        # fig, ax = plt.subplots(figsize=(4, 2), ncols=4, layout='constrained')
        # slc = (slice(9, 49), slice(10, 50), images.shape[-1]//2)
        # for i in range(4):
        #     ax[i].imshow(images[i+1][slc], vmin=0, vmax=1, cmap=CMAP['image'])
        #     ax[i].set_xticks([])
        #     ax[i].set_yticks([])
        # plt.savefig(path.join(save_dir, 'images.png'), dpi=300)
        # plt.show()
        # quit()
        
        psfs = []
        fwhms = []
        for i in range(1, len(images)):
            print('working on matrix shape {} with reference {}'.format(i, matrix_shapes[i], matrix_shapes[0]))
            psf_i, fwhm_i = map_resolution(images[0], images[i], patch_shape, resolution_mm, mask, args.stride, num_workers=args.workers)
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

        np.save(path.join(save_dir, 'images.npy'), images)
        np.save(path.join(save_dir, 'res-maps.npy'), fwhms)
        np.save(path.join(save_dir, 'res-masks.npy'), mask)
    
    else:

        with open(path.join(save_dir, 'args_post.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=4)

        data = np.load(path.join(save_dir, 'outputs.npz'))
        for var in data:
            globals()[var] = data[var]
    
    box_plots(fwhms / resolution_mm[0], matrix_shapes, save_dir=save_dir)

    plot_fwhm(fwhms / resolution_mm[0], (slice(None), slice(None), fwhms[0].shape[2]//2), save_dir=save_dir)

    volumes = []
    for i in range(2):
        for j in range(len(fwhms)):
            volumes += [fwhms[j][..., i] / resolution_mm[i]]
    fig, tracker = plotVolumes(volumes, figsize=(12, 4), nrows=2, ncols=len(fwhms), vmin=0, vmax=3, cmap=CMAP['resolution'], cbar=True)

    fig0, tracker0 = plotVolumes((images[0], images[1]), titles=('512x512', '256x256'))
    # fig1, tracker1 = plotVolumes((psfs[0, 0, 0, 0], psfs[0, 5, 5, 5]), vmin=0, vmax=3)
    # print(psfs.shape, fwhms.shape)
    # print(fwhms[0, 0, 0, 0], fwhms[0, 5, 5, 5])
    # psf = psfs[0, 0, 0, 0]
    # f = get_FWHM_from_pixel(psf)
    # print(f[0] * resolution_mm[0], f[1] * resolution_mm[1])
    # print('FWHM in pixels', f)
    
    if args.plot:
        plt.show()