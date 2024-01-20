import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from os import path, makedirs

from masks import get_mask_extrema, get_typical_level
from plot_artifact import plot_artifact_results, plot_progression
from plot import plotVolumes

from util import safe_divide, equalize, load_series, save_args

slc = (slice(40, 160), slice(65, 185), slice(10, 50))

p = argparse.ArgumentParser(description='Intensity artifact analysis of 2DFSE multi-slice image volumes with varying readout bandwidth.')
p.add_argument('root', type=str, help='path where outputs are saved')
p.add_argument('-e', '--exam_root', type=str, default=None, help='directory where exam data exists in subdirectories')
p.add_argument('-s', '--series_list', type=str, nargs='+', default=None, help='list of exam_root subdirectories to be analyzed in RBW-matched (plastic, metal) pairs')

if __name__ == '__main__':

    args = p.parse_args()
    save_dir = path.join(args.root, 'artifact')
    if not path.exists(save_dir):
        makedirs(save_dir)

    if args.exam_root is not None and args.series_list is not None:

        save_args(args, save_dir)

        images = [load_series(args.exam_root, series_name) for series_name in args.series_list]
        
        rbw = np.array([image.meta.readoutBandwidth_kHz for image in images[::2]])

        images = np.stack([image.data for image in images])

        images = equalize(images)
        
        signal_refs = np.stack([get_typical_level(image) for image in images[::2]])

        if slc is not None:
            images = images[(slice(None),) + slc]
            signal_refs = signal_refs[(slice(None),) + slc]

        maps_artifact = []
        for i in range(len(images) // 2):
            print('trial {} of {}'.format(i, len(images)//2))
            normalized_error = safe_divide(images[2*i+1] - images[2*i], signal_refs[i])
            _, map_artifact = get_mask_extrema(normalized_error, 0.3, 'mean', abs_margin=False)
            maps_artifact.append(map_artifact)
        maps_artifact = np.stack(maps_artifact)

        np.savez(path.join(save_dir, 'outputs.npz'),
            images=images,
            maps_artifact=maps_artifact,
            signal_refs=signal_refs,
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
        # figs[i], trackers[i] = plot_progression(images[2*i], images[2*i+1], maps_artifact[i], signal_refs[i])
        pass

    plot_artifact_results(images, maps_artifact, signal_refs, rbw, save_dir=save_dir)

    # fig, tracker = plotVolumes((images[0], signal_refs[0]), vmin=0.3, vmax=0.8)

    plt.show()