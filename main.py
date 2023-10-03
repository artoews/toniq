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
    
    # TODO check the ImageVolumes for isotropic resolution
    voxel_size_mm = target_image.meta.resolution_mm[0]

    if args.snr or args.resolution or args.geometric or args.intensity:
        map_all = False
    else:
        map_all = True
    
    mask_empty = analysis.get_mask_empty(clean_image.data)
    mask_signal = analysis.get_mask_signal(clean_image.data, target_image.data)
    
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
    
    if map_all or args.resolution:
        if args.verbose:
            print('Mapping resolution...')
            start_time = time()
        cell_size_pixels = args.cell_size_mm / voxel_size_mm
        patch_size = int(cell_size_pixels * 2)
        stride = int(cell_size_pixels / 2)
        psf_soln = psf.estimate_psf_all_in_parallel(clean_image.data, target_image.data, patch_size, stride, psf_size=7)
        fwhm = psf.get_FWHM_in_parallel(psf_soln)
        if args.verbose:
            print('Done. {:.1f} seconds elapsed.'.format(time() - start_time))
    
    if map_all or args.intensity:  # TODO cleanup!
        if args.verbose:
            print('Mapping intensity distortion...')
            start_time = time()
        mask_implant, mask_empty, mask_hyper, mask_hypo, mask_artifact = analysis.get_all_masks(clean_image.data, target_image.data)
        mask_artifact = morphology.dilation(mask_artifact, morphology.ball(2))
        mask_none = np.zeros_like(mask_implant)
        mask_to_artifact = analysis.combine_masks(mask_implant, mask_empty, mask_none, mask_none, mask_artifact)
        mask_to_hypo = analysis.combine_masks(mask_implant, mask_empty, mask_none, mask_hypo, mask_artifact)
        mask_to_all = analysis.combine_masks(mask_implant, mask_empty, mask_hyper, mask_hypo, mask_artifact)
        if args.verbose:
            print('Done. {:.1f} seconds elapsed.'.format(time() - start_time))

    if map_all or args.geometric:  # TODO cleanup into a function or two!
        if args.verbose:
            print('Mapping geometric distortion...')
            start_time = time()
        fixed_image = clean_image.data
        moving_image = target_image.data
        mega_mask = analysis.get_all_masks(fixed_image, moving_image, combine=True) # TODO re-use masks from above if available!
        fixed_mask = np.logical_not(analysis.get_mask_empty(fixed_image))
        fixed_mask = morphology.erosion(fixed_mask, morphology.ball(2))
        moving_mask = (mega_mask == 2/5)
        moving_mask = morphology.erosion(moving_mask, morphology.ball(2))
        signal_ref = analysis.get_typical_level(fixed_image)
        slc = (slice(25, 175), slice(50, 200), slice(10, 70))
        fixed_image = fixed_image[slc]
        fixed_mask = fixed_mask[slc]
        moving_image = moving_image[slc]
        moving_mask = moving_mask[slc]
        active_count = np.sum(moving_mask)
        total_count = np.prod(moving_mask.shape)
        print('{} of {} pixels active ({:.0f}%)'.format(active_count, total_count, active_count / total_count * 100))
        fixed_image_masked = fixed_image.copy()
        fixed_image_masked[~fixed_mask] = 0
        moving_image_masked = moving_image.copy()
        moving_image_masked[~moving_mask] = 0
        result, transform = register.nonrigid(fixed_image, moving_image, fixed_mask, moving_mask)
        result_masked = register.transform(moving_image_masked, transform)
        deformation_field = register.get_deformation_field(moving_image, transform)
        _, jacobian_det = register.get_jacobian(moving_image, transform)
        np.save(path.join(map_dir, 'deformation_field.npy'), deformation_field)
        if args.verbose:
            print('Done. {:.1f} seconds elapsed.'.format(time() - start_time))

    print('Target image', target_image.shape)

    # TODO in each block, save maps and optionally make a plot
