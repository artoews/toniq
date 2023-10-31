import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from os import path, makedirs
from pathlib import Path
import scipy.ndimage as ndi
from skimage import morphology
import sigpy as sp
from time import time
import analysis
import dicom
from plot import plotVolumes
import register
import distortion

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

    args = p.parse_args()
    
    # set up directory structure
    save_dir = path.join(args.root, 'distortion')
    if not path.exists(save_dir):
        makedirs(save_dir)
    
    slc = (slice(None), slice(25, 175), slice(50, 200), slice(10, 60))

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
        pbw = np.array([image.meta.pixelBandwidth_Hz for image in images[1:]])
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
        
        # TODO crop to lattice automatically, perhaps using get_mask_lattice ?? 
        images = images[slc]
        masks_register = masks_register[slc]

        # volumes = (images[0], images[1], images[2], masks_register[0], masks_register[1])
        # titles = ('plastic', 'fixed', 'moving', 'fixed mask', 'moving mask')
        # fig2, tracker2 = plotVolumes(volumes, titles=titles, figsize=(16, 8))
        # plt.show()
        # quit()
        
        # run registration
        print('Running registration...')
        results = []
        results_masked = []
        fields = []
        fixed_mask = masks_register[0]
        itk_parameters = register.setup_nonrigid()
        num_trials = len(images) - 2
        for i in range(num_trials):
            if args.verbose:
                print('on trial {} with PBWs {:.1f} and {:.1f} Hz'.format(i, pbw[0], pbw[1+i]))
            fixed_image = images[1]
            moving_image = images[2+i]
            moving_mask = masks_register[1+i]
            moving_image_masked = moving_image.copy()
            moving_image_masked[~moving_mask] = 0
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

        with open(path.join(save_dir, 'args_post.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=4)

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
    fixed_image = images[1]
    fixed_mask = masks_register[1]
    fixed_image_masked = fixed_image.copy()
    fixed_image_masked[~fixed_mask] = 0

    true_field = np.load(path.join(args.root, 'field', 'field.npy'))  # kHz
    true_field = -true_field / 2  # TODO hack
    true_field = ndi.median_filter(true_field, footprint=morphology.ball(5))
    true_field = true_field[slc[1:]] * 1000  # Hz
    fig4, tracker4 = plotVolumes((images[0], true_field / 12000 + 0.5), titles=('trial 0', 'true_field'))
    true_field_masked = true_field.copy()
    true_field_masked[~fixed_mask] = 0

    num_trials = len(images) - 2

    for i in range(num_trials):

        moving_image = images[2+i]
        moving_mask = masks_register[1+i]
        moving_image_masked = moving_image.copy()
        moving_image_masked[~moving_mask] = 0

        result = results[i]
        result_masked = results_masked[i]

        fields[i][~fixed_mask] = 0
        field_x = fields[i][..., 0]
        field_y = fields[i][..., 1]
        field_z = fields[i][..., 2]

        net_pbw = distortion.net_pixel_bandwidth(pbw[1+i], pbw[0])
        true_field_x = true_field_masked / net_pbw

        volumes = (true_field_x, field_x, field_x - true_field_x, field_x / true_field_x)
        titles = ('MSL field', 'Registration field', 'error')
        fig1, tracker1 = plotVolumes(volumes, titles=titles, figsize=(12, 8), cmap='RdBu', vmin=-10, vmax=10)

        volumes = (fixed_image, moving_image, result, fixed_image_masked, moving_image_masked, result_masked)
        titles = ('fixed', 'moving', 'result', 'fixed masked', 'moving masked', 'result masked')
        fig2, tracker2 = plotVolumes(volumes, 2, len(volumes) // 2, titles=titles, figsize=(16, 8))

        error = list(2 * np.abs(vol - fixed_image) * (np.abs(vol) > 0.1) for vol in volumes)
        fig3, tracker3 = plotVolumes(error, 2, len(volumes) // 2, titles=titles, figsize=(16, 8))

    print('pbw', pbw)

    # run stats, combining all results into one figure

    fig, ax = plt.subplots()
    f_max = 2000
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    for i in range(num_trials):
        if i == 0:
            continue
        net_pbw = distortion.net_pixel_bandwidth(pbw[1+i], pbw[0])
        print('pixel BWs', pbw[0], pbw[1+i], net_pbw)
        displacement_map = np.abs(fields[i][..., 0])
        # mask = (displacement_map > 1)  # TODO consider implant proximity masking instead
        disp = displacement_map.ravel()
        # true_field = disp * net_pbw  # hack for perfect result
        ax.scatter(np.abs(true_field_masked.ravel()), disp, c=colors[i], s=0.1, marker='.') # TODO compress information with random subsampling or replace scatter altogether with line plot error bands, e.g. https://seaborn.pydata.org/examples/errorband_lineplots.html
        ax.axline((0, 0), (f_max, f_max / net_pbw), color=colors[i], label='PBW={:.0f}Hz'.format(pbw[1+i]))
    ax.set_xlabel('field [Hz]')
    ax.set_ylabel('displacement [pixels]')
    ax.set_xlim([0, f_max])
    ax.set_ylim([0.5, f_max / net_pbw])

    plt.legend()

    plt.show()