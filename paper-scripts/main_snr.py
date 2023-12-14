import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from os import path, makedirs

import analysis
import plot_snr
from plot import plotVolumes

from util import equalize, load_series

# TODO systematize this
slc = (slice(40, 160), slice(65, 185), slice(15, 45))

p = argparse.ArgumentParser(description='Noise analysis of image volume duplicates.')
p.add_argument('root', type=str, help='path where outputs are saved')
p.add_argument('-e', '--exam_root', type=str, default=None, help='directory where exam data exists in subdirectories')
p.add_argument('-s', '--series_list', type=str, nargs='+', default=None, help='list of exam_root subdirectories to be analyzed, ordered by pairs')
p.add_argument('-c', '--unit_cell_mm', type=float, default=12.0, help='size of lattice unit cell (in mm)')
p.add_argument('-l', '--lattice_shape', type=int, nargs='+', default=[13, 13, 4], help='number of unit cells along each axis of lattice')
p.add_argument('-v', '--verbose', action='count', default=0, help='verbosity level')

if __name__ == '__main__':

    args = p.parse_args()
    save_dir = path.join(args.root, 'noise')
    if not path.exists(save_dir):
        makedirs(save_dir)

    if args.exam_root is not None and args.series_list is not None:

        with open(path.join(save_dir, 'args.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=4)
        
        images = [load_series(args.exam_root, series_name) for series_name in args.series_list]

        rbw = np.array([image.meta.readoutBandwidth_kHz for image in images[::2]])

        unit_cell_pixels = int(args.unit_cell_mm / images[0].meta.resolution_mm[0])

        images = np.stack([image.data for image in images])

        images = equalize(images)
          
        mask_empty = analysis.get_mask_empty(images[0])
        mask_signal = analysis.get_mask_signal(images[0])

        if slc is not None:
            images = images[(slice(None),) + slc]
            mask_empty = mask_empty[slc]
            mask_signal = mask_signal[slc]

        snrs = []
        signals = []
        noise_stds = []
        for i in range(0, len(images), 2):
            print('trial', i // 2)
            image1 = images[i]
            image2 = images[i+1]
            snr, signal, noise_std = analysis.signal_to_noise(image1, image2, mask_signal, mask_empty)
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
            mask_signal=mask_signal,
            rbw=rbw
         )

    else:
        with open(path.join(save_dir, 'args_post.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=4)
        data = np.load(path.join(save_dir, 'outputs.npz'))
        for var in data:
            globals()[var] = data[var]
    
    for i in range(0, len(images), 2):
        image1 = images[i]
        image2 = images[i+1]
        image_diff = 5 * (image2 - image1) + 0.5
        image_sum = 0.5 * (image2 + image1)
        volumes = (image1, image2, image_diff, image_sum)
        titles = ('Image 1', 'Image 2', 'Difference (5x)', 'Sum (0.5x)')
        fig, tracker = plotVolumes(volumes, 1, len(volumes), titles=titles, figsize=(16, 8))
    plot_snr.scatter(snrs, rbw, save_dir=save_dir)
    plot_snr.lines(snrs, rbw, save_dir=save_dir)
    plt.show()
