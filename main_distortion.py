import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from os import path, makedirs
from pathlib import Path
from time import time

import analysis
import dicom
from plot import plotVolumes
import register

from util import safe_divide


p = argparse.ArgumentParser(description='Geometric distortion analysis of 2DFSE multi-slice image volumes with varying readout bandwidth.')
p.add_argument('root', type=str, help='path where outputs are saved')
p.add_argument('-e', '--exam_root', type=str, default=None, help='directory where exam data exists in subdirectories')
p.add_argument('-s', '--series_list', type=str, nargs='+', default=None, help='list of exam_root subdirectories to be analyzed, with the first serving as plastic reference, second as metal reference')
p.add_argument('-v', '--verbose', action='count', default=0, help='verbosity level')

def load_dicom_series(path):
    if path is None:
        return None
    files = Path(path).glob('*MRDC*')
    image = dicom.load_series(files)
    return image

if __name__ == '__main__':

    # parse command line arguments and save state for reference
    args = p.parse_args()
    with open(path.join(args.root, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)
    
    # set up directory structure
    save_dir = path.join(args.root, 'distortion')
    if not path.exists(save_dir):
        makedirs(save_dir)
    
    if args.exam_root is not None and args.series_list is not None:

        # load data
        images = []
        for series_name in args.series_list:
            image = load_dicom_series(path.join(args.exam_root, series_name))
            images.append(image)
            if args.verbose:
                print('Found DICOM series {}; loaded data with shape {}'.format(series_name, image.shape))

        # extract relevant metadata and throw away the rest
        pbw = np.array([image.meta.pixelBandwidth_Hz for image in images])
        images = np.stack([image.data for image in images])
        
        # rescale data for comparison
        images[0] = analysis.normalize(images[0])
        for i in range(1, len(images)):
            images[i] = analysis.equalize(images[i], images[0])

        # compute masks
        if args.verbose:
            print('Computing masks...')
        mask_empty = analysis.get_mask_empty(images[0])
        mask_implant = analysis.get_mask_implant(mask_empty)
        mask_signal = analysis.get_mask_signal(images[0])
        signal_ref = analysis.get_typical_level(images[0], mask_signal, mask_implant)

        masks_register = []
        for image in images[1:]:
            mask_artifact = analysis.get_mask_artifact(image - images[0], signal_ref)
            mask_register = analysis.get_mask_register(mask_empty, mask_implant, mask_artifact)
            masks_register.append(mask_register)
        masks_register = np.stack(masks_register)
        
        # TODO crop to lattice automatically
        slc = (slice(None), slice(25, 175), slice(50, 200), slice(10, 60))
        images = images[slc]
        masks_register = masks_register[slc]

        # volumes = (images[0], images[1], images[2], masks_register[0], masks_register[1])
        # titles = ('plastic', 'fixed', 'moving', 'fixed mask', 'moving mask')
        # fig2, tracker2 = plotVolumes(volumes, titles=titles, figsize=(16, 8))
        # plt.show()
        # quit()
        
        # run registration
        if args.verbose:
            print('Running registration...')
        results = []
        results_masked = []
        fields = []
        fixed_mask = masks_register[0]
        itk_parameters = register.setup_nonrigid()
        for image, moving_mask in zip(images[2:], masks_register[1:]):
            fixed_image = images[1]
            moving_image = image 
            moving_image_masked = moving_image.copy()
            moving_image_masked[~moving_mask] = 0
            # result, transform = register.nonrigid(fixed_image, moving_image, fixed_mask, moving_mask)
            result, transform = register.elastix_registration(fixed_image, moving_image, fixed_mask, moving_mask, itk_parameters)
            field = register.get_deformation_field(moving_image, transform)
            result_masked = register.transform(moving_image_masked, transform)
            result_masked[np.abs(result_masked) < 1e-2] = 0  # TODO check this threshold
            result_masked[~fixed_mask] = 0
            results.append(result)
            results_masked.append(result_masked)
            fields.append(field)
        
        # save outputs
        if args.verbose:
            print('Saving outputs...')

        np.savez(path.join(save_dir, 'outputs.npz'),
                 images=images,
                 results=np.stack(results),
                 results_masked=np.stack(results_masked),
                 masks_register=np.stack(masks_register),
                 fields=np.stack(fields),
                 pbw=pbw
                 )
        # np.save(path.join(save_dir, 'images.npy'), images)
        # np.save(path.join(save_dir, 'results.npy'), np.stack(results))
        # np.save(path.join(save_dir, 'results_masked.npy'), np.stack(results_masked))
        # np.save(path.join(save_dir, 'masks_register.npy'), np.stack(masks_register))
        # np.save(path.join(save_dir, 'fields.npy'), np.stack(fields))
        # np.save(path.join(save_dir, 'pixelBandwidths_Hz.npy'), pbw)
        
    else:

        data = np.load(path.join(save_dir, 'outputs.npz'))
        for var in data:
            globals()[var] = data[var]
        
        # load outputs
        # images =          np.load(path.join(save_dir, 'images.npy'))
        # result =          np.load(path.join(save_dir, 'results.npy'))
        # result_masked =   np.load(path.join(save_dir, 'results_masked.npy'))
        # masks_register =  np.load(path.join(save_dir, 'masks_register.npy'))
        # fields =          np.load(path.join(save_dir, 'fields.npy'))
        # pbw =             np.load(path.join(save_dir, 'pixelBandwidths_Hz.npy'))

    # plot image results figure for each trial
    i = 0  # trial index
    fixed_image = images[1]
    moving_image = images[2+i]
    fixed_mask = masks_register[1]
    moving_mask = masks_register[1+i]
    result = results[i]
    result_masked = results_masked[i]

    fixed_image_masked = fixed_image.copy()
    fixed_image_masked[fixed_mask] = 0
    moving_image_masked = moving_image.copy()
    moving_image_masked[moving_mask] = 0

    fields[i][np.abs(fields[i]) < 0.5] = 0
    field_x = fields[i][..., 0] / 10 + 0.5
    field_y = fields[i][..., 1] / 10 + 0.5
    field_z = fields[i][..., 2] / 10 + 0.5

    volumes = (field_x, field_y, field_z)
    titles = ('field x', 'field y', 'field z')
    fig1, tracker1 = plotVolumes(volumes, titles=titles, figsize=(12, 8), cmap='RdBu')

    volumes = (fixed_image, moving_image, result, fixed_image_masked, moving_image_masked, result_masked)
    titles = ('fixed', 'moving', 'result', 'fixed masked', 'moving masked', 'result masked')
    fig2, tracker2 = plotVolumes(volumes, 2, len(volumes) // 2, titles=titles, figsize=(16, 8))

    error = list(2 * np.abs(vol - fixed_image) * (np.abs(vol) > 0.1) for vol in volumes)
    fig3, tracker3 = plotVolumes(error, 2, len(volumes) // 2, titles=titles, figsize=(16, 8))

    print('pbw', pbw)

    plt.show()

    # run stats, combining all results into one figure

        # group voxels by distance to the registration masks
        # displacement vs b0 plot with a separate colr for each distortion instance
                # reference line for given RBW
                # data for each group represented by a line with error bands