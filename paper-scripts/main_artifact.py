import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from os import path, makedirs

from masks import get_mask_extrema, get_typical_level
from plot_artifact import plot_artifact_results, plot_artifact_results_overlay, plot_progression

from util import safe_divide, equalize, load_series

slc = (slice(40, 160), slice(65, 185), slice(10, 50))

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

        maps_artifact = []
        for i in range(1, len(images)):
            print('trial', i)
            normalized_error = safe_divide(images[i] - images[0], signal_ref)
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
    figs = [None] * num_trials
    trackers = [None] * num_trials
    for i in range(num_trials):
        figs[i], trackers[i] = plot_progression(images[0], images[1+i], maps_artifact[i], signal_ref)

    plot_artifact_results(images, maps_artifact, signal_ref, rbw, save_dir=save_dir)
    # plot_artifact_results_overlay(images, maps_artifact, signal_ref, rbw, save_dir=save_dir)

    plt.show()