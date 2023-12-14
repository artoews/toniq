import argparse
import json
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from os import path, makedirs

from masks import get_mask_extrema
from intensity import get_typical_level
from plot import plotVolumes

from util import safe_divide, equalize, load_series

# TODO systematize this
slc = (slice(35, 155), slice(65, 185), slice(15, 45))

p = argparse.ArgumentParser(description='Intensity artifact analysis of 2DFSE multi-slice image volumes with varying readout bandwidth.')
p.add_argument('root', type=str, help='path where outputs are saved')
p.add_argument('-e', '--exam_root', type=str, default=None, help='directory where exam data exists in subdirectories')
p.add_argument('-s', '--series_list', type=str, nargs='+', default=None, help='list of exam_root subdirectories to be analyzed, with the first serving as plastic reference')

if __name__ == '__main__':

    args = p.parse_args()
    save_dir = path.join(args.root, 'artifact')
    if not path.exists(save_dir):
        makedirs(save_dir)

    if args.exam_root is not None and args.series_list is not None:

        with open(path.join(save_dir, 'args.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=4)

        images = [load_series(args.exam_root, series_name) for series_name in args.series_list]
        
        rbw = np.array([image.meta.readoutBandwidth_kHz for image in images[1:]])

        images = np.stack([image.data for image in images])

        images = equalize(images)
        
        signal_ref = get_typical_level(images[0])

        if slc is not None:
            images = images[(slice(None),) + slc]
            signal_ref = signal_ref[slc]

        num_trials = len(images) - 1
        maps_artifact = []

        for i in range(num_trials):
            print('trial', i)
            image_ref = images[0]
            image_i = images[1+i]
            error_i = image_i - image_ref
            normalized_error = safe_divide(error_i, signal_ref)
            _, map_artifact = get_mask_extrema(normalized_error, 0.3, 'mean', abs_margin=False)
            maps_artifact.append(map_artifact)
        maps_artifact = np.stack(maps_artifact)

        np.savez(path.join(save_dir, 'outputs.npz'),
            images=images,
            maps_artifact=maps_artifact,
            signal_ref=signal_ref,
            rbw=rbw
         )
    
    else:
        with open(path.join(save_dir, 'args_post.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=4)
        data = np.load(path.join(save_dir, 'outputs.npz'))
        for var in data:
            globals()[var] = data[var]
    
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
    fig, axes = plt.subplots(nrows=num_trials, ncols=6, figsize=(18, 6), gridspec_kw={'width_ratios': [1, 1, 1, 1, 1, 0.1]})
    if num_trials == 1:
        axes = axes[None, :]
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0, 0].imshow(image_ref[slc], cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('Plastic', fontsize=fs)
    axes[0, 1].set_title('Metal', fontsize=fs)
    axes[0, 2].set_title('Relative Error', fontsize=fs)
    axes[0, 3].set_title('+ Mean Filter', fontsize=fs)
    axes[0, 4].set_title('+ Threshold (30%)', fontsize=fs)
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
        im2 = axes[i, 2].imshow(normalized_error_i[slc], cmap='RdBu_r', vmin=-1, vmax=1)
        axes[i, 3].imshow(maps_artifact[i][slc], cmap='RdBu_r', vmin=-1, vmax=1)
        axes[i, 4].imshow(mask_artifact[slc], cmap='RdBu_r', vmin=-1, vmax=1)
        axes[i, 1].set_ylabel('RBW={:.3g}kHz'.format(rbw[i]), fontsize=fs)
        plt.colorbar(im2, cax=axes[i, 5], ticks=[-1, 0, 1])
        axes[i, 5].tick_params(labelsize=fs*0.75)
    plt.savefig(path.join(save_dir, 'validation_artifact.png'), dpi=300)
    plt.show()