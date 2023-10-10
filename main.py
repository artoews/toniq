import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from os import path, makedirs
from pathlib import Path
from skimage import morphology
from time import time

import analysis
import dicom
import fwhm as fwh
from plot import plotVolumes
import psf
import register

# TODO use image resolution, or lattice cell size (?) to determine the filter size for operations with filter radius > 2

p = argparse.ArgumentParser(description='Image quality mapping toolbox for image volumes from metal phantom scans')
p.add_argument('out', type=str, help='path where outputs are saved')
p.add_argument('target_image', type=str, help='path to image volume; target for analysis')
p.add_argument('clean_image', type=str, default=None, help='path to image volume; reference for analysis')
p.add_argument('-d', '--repeat_image', type=str, default=None, help='path to image volume; repetition of target; default=None')
p.add_argument('-c', '--cell_size_mm', type=float, default=12, help='size of lattice unit cell (in mm); default=12')
p.add_argument('-s', '--snr', action='store_true', help='map SNR, and nothing else unless explicitly indicated')
p.add_argument('-r', '--resolution', action='store_true', help='map resolution, and nothing else unless explicitly indicated')
p.add_argument('-i', '--intensity', action='store_true', help='map intensity distortion, and nothing else unless explicitly indicated')
p.add_argument('-g', '--geometric', action='store_true', help='map geometric distortion, and nothing else unless explicitly indicated')
p.add_argument('-n', '--num_workers', type=int, default=8, help='number of workers used for parallelized tasks (mapping resolution); default=8')
p.add_argument('-p', '--plot', action='store_true', help='make plots and show')
p.add_argument('-v', '--verbose', action='count', default=0, help='verbosity level')


def setup_dirs(root):
    map_dir = path.join(root, 'map')
    plot_dir = path.join(root, 'plot')
    dirs = (map_dir, plot_dir)
    for d in dirs:
        if not path.exists(d):
            makedirs(d)
    return dirs 

def load_dicom_series(path):
    if path is None:
        return None
    files = Path(path).glob('*MRDC*')
    image = dicom.load_series(files)
    return image


if __name__ == '__main__':
    args = p.parse_args()
    with open(path.join(args.out, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    map_dir, plot_dir = setup_dirs(args.out)

    target_image = load_dicom_series(args.target_image)
    clean_image = load_dicom_series(args.clean_image)
    repeat_image = load_dicom_series(args.repeat_image)

    clean_image.data = analysis.normalize(clean_image.data)
    target_image.data = analysis.equalize(target_image.data, clean_image.data)
    if repeat_image is not None:
        repeat_image.data = analysis.equalize(repeat_image.data, clean_image.data)

    print('Clean data shape', clean_image.data.shape)
    print('Target data shape', target_image.data.shape)
    
    if target_image.is_isotropic:
        voxel_size_mm = target_image.meta.resolution_mm[0]
    else:
        raise ValueError('Isotropic resolution is required, but got: ', target_image.meta.resolution_mm)

    if args.snr or args.resolution or args.geometric or args.intensity:
        map_all = False
    else:
        map_all = True
    
    mask_empty = analysis.get_mask_empty(clean_image.data)
    mask_signal = analysis.get_mask_signal(clean_image.data, target_image.data)

    if args.plot:
        volumes = (clean_image.data, target_image.data)
        titles = ('clean image', 'target image')
        fig0, tracker0 = plotVolumes(volumes, titles=titles, figsize=(16, 8))

    
    if map_all or args.snr:

        if repeat_image is None:
            if args.verbose:
                print('Skipping SNR map, no repeat image provided')
        else:
            if args.verbose:
                print('Mapping SNR...')
                start_time = time()

            snr, signal, noise_std = analysis.signal_to_noise(clean_image.data, target_image.data, mask_signal, mask_empty)

            if args.verbose:
                print('Done. {:.1f} seconds elapsed.'.format(time() - start_time))

            np.save(path.join(map_dir, 'snr.npy'), snr)
            np.save(path.join(map_dir, 'signal.npy'), signal)
            np.save(path.join(map_dir, 'noise.npy'), noise_std)
    
    if map_all or args.resolution:

        if args.verbose:
            print('Mapping resolution...')
            start_time = time()

        cell_size_pixels = args.cell_size_mm / voxel_size_mm
        patch_shape = (int(cell_size_pixels),) * 3
        psf_shape = (5, 5, 1)
        stride = int(cell_size_pixels / 2)
        # clean_input = analysis.denoise(clean_image.data)
        # target_input = analysis.denoise(target_image.data)
        clean_input = clean_image.data[..., 30:48]
        target_input = target_image.data[..., 30:48]
        psf_soln = psf.map_psf(clean_input, target_input, patch_shape, psf_shape, stride, 'iterative', mask=None, num_workers=8)
        print('psf_soln', psf_soln.shape)
        start_fwhm_time = time()
        fwhm = fwh.get_FWHM_in_parallel(psf_soln)
        resolution = fwhm * voxel_size_mm  # TODO what if clean and target resolution are different?
        print('FWHM time: {:.1f} seconds elapsed.'.format(time() - start_fwhm_time))

        if args.verbose:
            print('Done. {:.1f} seconds elapsed.'.format(time() - start_time))

        np.save(path.join(map_dir, 'resolution.npy'), resolution)

        if args.plot:
            # psf_slc = np.abs(psf_soln[8, 8, 2, ...])
            psf_slc = np.moveaxis(np.abs(psf_soln[15, 15, ...]), 0, -1)
            psf_slc = psf_slc / np.max(psf_slc)
            volumes = (psf_slc, psf_slc)
            print('FWHM shape', fwhm.shape)
            titles = ('PSF with FWHM {} pixels'.format(fwhm[15, 15, 3, :]), 'Same')
            # fig1, tracker1 = plotVolumes(volumes, titles=titles, figsize=(16, 8))
            # volumes = (fwhm[..., 0], fwhm[..., 1], fwhm[..., 2])
            # titles = ('FWHM x [mm]', 'FWHM y [mm]', 'FWHM z [mm]')
            volumes = (fwhm[..., 0], fwhm[..., 1])
            titles = ('FWHM x [pixels]', 'FWHM y [pixels]')
            fig2, tracker2 = plotVolumes(volumes, titles=titles, figsize=(16, 8), vmin=0, vmax=10, cmap='tab20c', cbar=True)

    if map_all or args.intensity or args.geometric:

        if args.verbose:
            print('Mapping intensity distortion...')
            start_time = time()

        mask_implant, mask_empty, mask_hyper, mask_hypo, mask_artifact = analysis.get_all_masks(clean_image.data, target_image.data)
        # mask_artifact = morphology.dilation(mask_artifact, morphology.ball(2))

        if args.verbose:
            print('Done. {:.1f} seconds elapsed.'.format(time() - start_time))

        np.save(path.join(map_dir, 'mask_implant.npy'), mask_implant)
        np.save(path.join(map_dir, 'mask_empty.npy'), mask_empty)
        np.save(path.join(map_dir, 'mask_hyper.npy'), mask_hyper)
        np.save(path.join(map_dir, 'mask_hypo.npy'), mask_hypo)
        np.save(path.join(map_dir, 'mask_artifact.npy'), mask_artifact)

    if map_all or args.geometric:

        if args.verbose:
            print('Mapping geometric distortion...')
            start_time = time()

        fixed_image = clean_image.data
        moving_image = target_image.data
        mega_mask = analysis.combine_masks(mask_implant, mask_empty, mask_hyper, mask_hypo, mask_artifact)
        
        fixed_mask = np.logical_not(mask_empty)
        fixed_mask = morphology.erosion(fixed_mask, morphology.ball(2))
        moving_mask = (mega_mask == 2/5)
        moving_mask = morphology.erosion(moving_mask, morphology.ball(2))

        slc = (slice(25, 175), slice(50, 200), slice(10, 70))
        fixed_image = fixed_image[slc]
        fixed_mask = fixed_mask[slc]
        moving_image = moving_image[slc]
        moving_mask = moving_mask[slc]

        active_count = np.sum(moving_mask)
        total_count = np.prod(moving_mask.shape)

        if args.verbose:
            print('{} of {} pixels active ({:.0f}%)'.format(active_count, total_count, active_count / total_count * 100))

        deformation_field, jacobian_det, result, result_masked = analysis.estimate_geometric_distortion(fixed_image, moving_image, fixed_mask, moving_mask)

        if args.verbose:
            print('Done. {:.1f} seconds elapsed.'.format(time() - start_time))

        np.save(path.join(map_dir, 'deformation_field.npy'), deformation_field)
        np.save(path.join(map_dir, 'jacobian_det.npy'), deformation_field)
        np.save(path.join(map_dir, 'registration_result.npy'), result)
        np.save(path.join(map_dir, 'registration_result_masked.npy'), result_masked)
    
    plt.show()
