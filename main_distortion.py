import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from os import path, makedirs
from pathlib import Path
import scipy.ndimage as ndi
from skimage import morphology
import seaborn as sns
import sigpy as sp
from time import time
import analysis
import dicom
from plot import plotVolumes, overlay_mask
import register
import distortion

from util import safe_divide, masked_copy


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
    
    # slc = (slice(None), slice(25, 175), slice(50, 200), slice(10, 60))
    slc = (slice(None), slice(35, 155), slice(65, 185), 30)

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
        rbw = np.array([image.meta.readoutBandwidth_kHz for image in images[1:]])
        images = np.stack([image.data for image in images])
        print('pbw', pbw)
        print('rbw', rbw)
        
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
            error = image - images[0]
            normalized_error = safe_divide(error, signal_ref)
            mask_artifact = analysis.get_mask_artifact(normalized_error)
            mask_register = analysis.get_mask_register(mask_empty, mask_implant, mask_artifact)
            masks_register.append(mask_register)
        masks_register = np.stack(masks_register)
        
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
        deformation_fields = []
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
            deformation_field = register.get_deformation_field(moving_image, transform)
            result_masked = register.transform(moving_image_masked, transform)
            result_mask = np.logical_and(np.abs(result_masked) > 0.1, fixed_mask)
            result_masked = masked_copy(result_masked, result_mask)
            results.append(result)
            results_masked.append(result_masked)
            deformation_fields.append(deformation_field)
        
        # save outputs
        if args.verbose:
            print('Saving outputs...')

        np.savez(path.join(save_dir, 'outputs.npz'),
                 images=images,
                 results=np.stack(results),
                 results_masked=np.stack(results_masked),
                 masks_register=np.stack(masks_register),
                 deformation_fields=np.stack(deformation_fields),
                 pbw=pbw,
                 rbw=rbw
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

    # plot image results figure for each trial
    fixed_image = images[1]
    fixed_mask = masks_register[1]
    fixed_image_masked = masked_copy(fixed_image, fixed_mask)

    # true_field = np.load(path.join(args.root, 'field', 'field.npy'))  # kHz
    true_field = np.load(path.join(args.root, 'field', 'field-metal.npy')) - np.load(path.join(args.root, 'field', 'field-plastic.npy'))  # kHz
    true_field = -true_field
    true_field = ndi.median_filter(true_field, footprint=morphology.ball(4))
    # true_field = ndi.generic_filter(true_field, np.mean, footprint=morphology.ball(3))
    true_field = true_field[slc[1:]] * 1000  # Hz

    # fig4, tracker4 = plotVolumes((images[0] * 24e3 - 12e3, true_field), titles=('trial 0', 'true_field'), vmin=-12e3, vmax=12e3)
    # true_field_masked = masked_copy(true_field, fixed_mask)

    num_trials = len(images) - 2

    # abstract validation figure panel A: image result

    fig, axes = plt.subplots(nrows=num_trials, ncols=5, figsize=(20, 8))
    fs = 20
    kwargs = {'cmap': 'gray', 'vmin': 0, 'vmax': 1}
    axes[0, 0].imshow(fixed_image_masked, **kwargs)
    overlay_mask(axes[0, 0], ~fixed_mask)
    error_multiplier = 3
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    for i in range(num_trials):
        moving_mask = masks_register[1+i]
        moving_image_masked = masked_copy(images[2+i], moving_mask)
        init_error = np.abs(moving_image_masked - fixed_image_masked) * error_multiplier
        result_error = np.abs(results_masked[i] - fixed_image_masked) * error_multiplier
        init_mask =  (moving_image_masked != 0) * (fixed_image_masked != 0)
        result_mask = (results_masked[i] != 0)
        # color_mask = np.zeros(result_error.shape + (4,), dtype=np.uint8)
        # color_mask[~result_mask, :] = np.array([0, 0, 255, 255], dtype=np.uint8)
        axes[i, 1].imshow(moving_image_masked, **kwargs)
        axes[i, 2].imshow(results_masked[i], **kwargs)
        axes[i, 3].imshow(init_error * init_mask, **kwargs)
        axes[i, 4].imshow(result_error * result_mask, **kwargs)
        overlay_mask(axes[i, 1], ~moving_mask)
        overlay_mask(axes[i, 2], ~result_mask)
        overlay_mask(axes[i, 3], ~init_mask)
        overlay_mask(axes[i, 4], ~result_mask)
        axes[i, 1].set_ylabel('RBW={:.3g}kHz'.format(rbw[1+i]), fontsize=fs)
        if i > 0:
            plt.delaxes(axes[i, 0])
    axes[0, 0].set_title('Fixed Image', fontsize=fs)
    axes[0, 1].set_title('Moving Image', fontsize=fs)
    axes[0, 2].set_title('Registration', fontsize=fs)
    axes[0, 3].set_title('Initial Error (3x)', fontsize=fs)
    axes[0, 4].set_title('Final Error (3x)', fontsize=fs)
    axes[0, 0].set_ylabel('RBW={:.3g}kHz'.format(rbw[0]), fontsize=fs)
    plt.savefig(path.join(save_dir, 'validation_distortion_images.png'), dpi=300)

    # abstract validation figure panel B: field result

    fig, axes = plt.subplots(nrows=num_trials, ncols=4, figsize=(12, 8), gridspec_kw={'width_ratios': [1, 1, 1, 0.1]})
    kwargs = {'cmap': 'RdBu', 'vmin': -4, 'vmax': 4}
    fs = 20
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    for i in range(num_trials):
        net_pbw = distortion.net_pixel_bandwidth(pbw[1+i], pbw[0])
        result_mask = (results_masked[i] != 0)
        moving_image_masked = masked_copy(images[2+i], masks_register[1+i])
        simulated_deformation = true_field / net_pbw
        measured_deformation = deformation_fields[i][..., 0]
        axes[i, 0].imshow(simulated_deformation * result_mask, **kwargs)
        axes[i, 1].imshow(measured_deformation * result_mask, **kwargs)
        im = axes[i, 2].imshow((simulated_deformation - measured_deformation) * result_mask, **kwargs)
        overlay_mask(axes[i, 0], ~result_mask)
        overlay_mask(axes[i, 1], ~result_mask)
        overlay_mask(axes[i, 2], ~result_mask)
        axes[i, 0].set_ylabel('RBW={:.3g}kHz'.format(rbw[1+i]), fontsize=fs)
        cb = plt.colorbar(im, cax=axes[i, 3], ticks=[-4, -2, 0, 2, 4])
        cb.set_label(label='Readout Disp. (pixels)', size=int(fs*0.7))
    axes[0, 0].set_title('Reference', fontsize=fs)
    axes[0, 1].set_title('Registration', fontsize=fs)
    axes[0, 2].set_title('Error', fontsize=fs)
    plt.savefig(path.join(save_dir, 'validation_distortion_fields.png'), dpi=300)

    # abstract validation figure panel C: line plots

    fig, ax = plt.subplots(figsize=(8, 5))
    f_max = 1500
    fs = 14
    colors = ['black', 'red', 'blue']
    for i in range(num_trials):
        result_mask = (results_masked[i] != 0)
        net_pbw = distortion.net_pixel_bandwidth(pbw[1+i], pbw[0])
        measured_deformation = deformation_fields[i][..., 0]
        field_bins = np.round(true_field / 100) * 100
        # measured_deformation = np.abs(measured_deformation)
        # field_bins = np.abs(field_bins)
        # plots mean line and 95% confidence band
        sns.lineplot(x=(field_bins * result_mask).ravel(),
                     y=(measured_deformation * result_mask).ravel(),
                     ax=ax, legend='brief', label='{0:.3g}kHz'.format(rbw[i+1]), color=colors[i])
        # ax.scatter((field_bins * result_mask).ravel(), (np.abs(measured_deformation) * result_mask).ravel(), c=colors[i], s=0.1, marker='.')
        ax.axline((-f_max, -f_max / net_pbw), (f_max, f_max / net_pbw), color=colors[i], linestyle='--')
    ax.set_xlabel('Off-Resonance (Hz)', fontsize=fs)
    ax.set_ylabel('Readout Disp. (pixels)', fontsize=fs)
    ax.set_xlim([-f_max, f_max])
    ax.set_ylim([-f_max / net_pbw, f_max / net_pbw])
    plt.legend(title='Readout BW', fontsize=fs)
    plt.grid()
    plt.savefig(path.join(save_dir, 'validation_distortion_summary.png'), dpi=300)

    plt.show()