import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from os import path, makedirs

import plot_snr

from masks import get_signal_mask, get_artifact_mask
from intensity import map_snr
from plot import plotVolumes

from util import equalize, load_series, save_args
from slice_params import *

slc = LATTICE_SLC

p = argparse.ArgumentParser(description='Noise analysis of image volume duplicates.')
p.add_argument('root', type=str, help='path where outputs are saved')
p.add_argument('-e', '--exam_root', type=str, default=None, help='directory where exam data exists in subdirectories')
p.add_argument('-s', '--series_list', type=str, nargs='+', default=None, help='list of exam_root subdirectories to be analyzed, ordered by pairs')
p.add_argument('-c', '--unit_cell_mm', type=float, default=12.0, help='size of lattice unit cell (in mm); default=12')
p.add_argument('-t', '--threshold', type=float, default=None, help='maximum intensity artifact error included in mask; default=None')
p.add_argument('-p', '--plot', action='store_true', help='show plots')

if __name__ == '__main__':

    args = p.parse_args()
    save_dir = path.join(args.root, 'snr')
    artifact_dir = path.join(args.root, 'artifact')
    if not path.exists(save_dir):
        makedirs(save_dir)

    if args.exam_root is not None and args.series_list is not None:
        save_args(args, save_dir) 
        images = [load_series(args.exam_root, series_name) for series_name in args.series_list]
        num_trials = len(images) // 2
        rbw = np.array([image.meta.readoutBandwidth_kHz for image in images[::2]])
        unit_cell_pixels = int(args.unit_cell_mm / images[0].meta.resolution_mm[0])
        images = np.stack([image.data for image in images])
        images = equalize(images)
          
        images = images[(slice(None),) + slc]

        implant_mask = np.load(path.join(artifact_dir, 'implant-mask.npy'))
        if args.threshold is None:
            artifact_maps = None
            masks = [get_signal_mask(implant_mask)] * num_trials
        else:
            ia_maps = np.load(path.join(artifact_dir, 'ia-maps.npy'))
            artifact_masks = [get_artifact_mask(ia_map, args.threshold) for ia_map in ia_maps]
            masks = [get_signal_mask(implant_mask, artifact_masks=[artifact_mask]) for artifact_mask in artifact_masks]
        
        snrs = []
        signals = []
        noise_stds = []
        for i in range(num_trials):
            print('trial', i)
            snr, signal, noise_std = map_snr(images[2*i], images[2*i+1], masks[i])
            snrs.append(snr)
            signals.append(signal)
            noise_stds.append(noise_std)
        snrs = np.stack(snrs)
        signals = np.stack(signals)
        noise_stds = np.stack(noise_stds)

        np.savez(path.join(save_dir, 'outputs.npz'),
            images=images,
            snrs=snrs,
            signals=signals,
            noise_stds=noise_stds,
            rbw=rbw
         )
        
        np.save(path.join(save_dir, 'images.npy'), images)
        np.save(path.join(save_dir, 'snr-maps.npy'), snrs)
        np.save(path.join(save_dir, 'snr-masks.npy'), masks)

    else:
        with open(path.join(save_dir, 'args_post.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=4)
        data = np.load(path.join(save_dir, 'outputs.npz'))
        for var in data:
            globals()[var] = data[var]
    
    figs = [None,] * (len(images)//2)
    trackers = [None,] * (len(images)//2)
    for i in range(0, len(images), 2):
        image1 = images[i]
        image2 = images[i+1]
        image_diff = 5 * (image2 - image1) + 0.5
        image_sum = 0.5 * (image2 + image1)
        volumes = (image1, image2, image_diff, image_sum, snrs[i//2] / 60, noise_stds[i//2] * 50)
        titles = ('Image 1', 'Image 2', 'Difference (5x)', 'Sum (0.5x)', 'SNR / 60', 'Noise STD * 50')
        # figs[i//2], trackers[i//2] = plotVolumes(volumes, 1, len(volumes), titles=titles, figsize=(16, 8))
    plot_snr.scatter(snrs, rbw, save_dir=save_dir)
    plot_snr.lines(snrs, rbw, save_dir=save_dir)
    fig2, tracker2 = plotVolumes((images[0], images[2]), titles=("{}kHz".format(rbw[0]), "{}kHz".format(rbw[1])))
    fig3, tracker3 = plotVolumes((snrs[0], snrs[1]), vmax=150, cmap='viridis', cbar=True)

    if args.plot:
        plt.show()
