import argparse
import json
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from os import path, makedirs
from pathlib import Path
import seaborn as sns
from time import time

import analysis
import dicom
from plot import plotVolumes

from util import safe_divide


p = argparse.ArgumentParser(description='Intensity artifact analysis of 2DFSE multi-slice image volumes with varying readout bandwidth.')
p.add_argument('root', type=str, help='path where outputs are saved')
p.add_argument('-e', '--exam_root', type=str, default=None, help='directory where exam data exists in subdirectories')
p.add_argument('-s', '--series_list', type=str, nargs='+', default=None, help='list of exam_root subdirectories to be analyzed, with the first serving as plastic reference')
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
    save_dir = path.join(args.root, 'artifact')
    if not path.exists(save_dir):
        makedirs(save_dir)

    slc = (slice(35, 155), slice(65, 185), slice(15, 45))

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
        rbw = np.array([image.meta.readoutBandwidth_kHz for image in images[1:]])
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

        images = images[(slice(None),) + slc]
        mask_empty = mask_empty[slc]
        mask_implant = mask_implant[slc]
        mask_signal = mask_signal[slc]
        signal_ref = signal_ref[slc]

        num_trials = len(images) - 1
        maps_artifact = []

        # map intensity artifact
        if args.verbose:
            print('Computing artifact map...')
        for i in range(num_trials):
            print('trial', i)
            image_ref = images[0]
            image_i = images[1+i]
            error_i = image_i - image_ref
            normalized_error = safe_divide(error_i, signal_ref)
            _, map_artifact = analysis.get_mask_extrema(normalized_error, 0.3, 'mean', abs_margin=False)
            maps_artifact.append(map_artifact)

        # save outputs
        if args.verbose:
            print('Saving outputs...')

        np.savez(path.join(save_dir, 'outputs.npz'),
            images=images,
            maps_artifact=np.stack(maps_artifact),
            mask_signal=mask_signal,
            signal_ref=signal_ref,
            rbw=rbw
         )

        print('done saving outputs')
    
    else:

        with open(path.join(save_dir, 'args_post.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=4)

        data = np.load(path.join(save_dir, 'outputs.npz'))
        for var in data:
            globals()[var] = data[var]
    
    if args.verbose:
        print('Begin plotting...')

    num_trials = len(images) - 1
    image_ref = images[0]

    figs = [None] * num_trials
    trackers = [None] * num_trials
    for i in range(num_trials):
        continue
        image_i = images[1+i]
        error_i = image_i - image_ref
        normalized_error_i = safe_divide(error_i, signal_ref)
        map_artifact_i = maps_artifact[i]
        mask_artifact = np.zeros_like(map_artifact_i)
        mask_artifact[map_artifact_i > 0.3] = 0.3
        mask_artifact[map_artifact_i < -0.3] = -0.3

        volumes = (image_ref - 0.5,
                   image_i - 0.5,
                   error_i,
                   normalized_error_i,
                   map_artifact_i,
                   mask_artifact
                   )

        titles = ('plastic', 'metal', 'diff', 'norm. diff.', 'filtered norm. diff.', 'threshold at +/-30%')
        figs[i], trackers[i] = plotVolumes(volumes, 1, len(volumes), titles=titles, figsize=(16, 8), vmin=-1, vmax=1)
    
    # make abstract figure
    slc = (slice(None), slice(None), image_ref.shape[2] // 2)
    fs = 20
    fig, axes = plt.subplots(nrows=num_trials, ncols=7, figsize=(18, 6), gridspec_kw={'width_ratios': [1, 1, 0.1, 1, 1, 1, 0.1]})
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0, 0].imshow(image_ref[slc], cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('Plastic', fontsize=fs)
    axes[0, 1].set_title('Metal', fontsize=fs)
    axes[0, 3].set_title('Relative Error', fontsize=fs)
    axes[0, 4].set_title('+ Mean Filter', fontsize=fs)
    axes[0, 5].set_title('+ Threshold (30%)', fontsize=fs)
    # axes[0, 3].set_title('Intensity Artifact Map')
    # axes[0, 4].set_title('Intensity Artifact Mask')
    for i in range(num_trials):
        image_i = images[1+i]
        error_i = image_i - image_ref
        normalized_error_i = safe_divide(error_i, signal_ref)
        mask_artifact = np.zeros_like(maps_artifact[i])
        mask_artifact[maps_artifact[i] > 0.3] = 0.3
        mask_artifact[maps_artifact[i] < -0.3] = -0.3
        if i > 0: plt.delaxes(axes[i, 0])
        im1 = axes[i, 1].imshow(image_i[slc], cmap='gray', vmin=0, vmax=1)
        im2 = axes[i, 3].imshow(normalized_error_i[slc], cmap='RdBu', vmin=-1, vmax=1)
        axes[i, 4].imshow(maps_artifact[i][slc], cmap='RdBu', vmin=-1, vmax=1)
        axes[i, 5].imshow(mask_artifact[slc], cmap='RdBu', vmin=-1, vmax=1)
        axes[i, 1].set_ylabel('RBW={:.3g}kHz'.format(rbw[i]), fontsize=fs)
        plt.colorbar(im1, cax=axes[i, 2], ticks=[0, 1])
        plt.colorbar(im2, cax=axes[i, 6], ticks=[-1, 0, 1])
    plt.savefig(path.join(save_dir, 'artifact_validation.png'))
    plt.show()