import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from os import path, makedirs
from pathlib import Path
import seaborn as sns
from time import time

import analysis
import dicom
from plot import plotVolumes

from util import safe_divide


p = argparse.ArgumentParser(description='Noise analysis of image volume duplicates.')
p.add_argument('root', type=str, help='path where outputs are saved')
p.add_argument('-e', '--exam_root', type=str, default=None, help='directory where exam data exists in subdirectories')
p.add_argument('-s', '--series_list', type=str, nargs='+', default=None, help='list of exam_root subdirectories to be analyzed, ordered by pairs')
p.add_argument('-c', '--unit_cell_mm', type=float, default=12.0, help='size of lattice unit cell (in mm)')
p.add_argument('-l', '--lattice_shape', type=int, nargs='+', default=[13, 13, 4], help='number of unit cells along each axis of lattice')
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
    save_dir = path.join(args.root, 'noise')
    if not path.exists(save_dir):
        makedirs(save_dir)

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
        
        if images[0].is_isotropic:
            voxel_size_mm = images[0].meta.resolution_mm[0]
        else:
            raise ValueError('Isotropic resolution is required, but got: ', images[0].meta.resolution_mm)
        unit_cell_pixels = int(args.unit_cell_mm / voxel_size_mm)

        # extract relevant metadata and throw away the rest
        rbw = np.array([image.meta.readoutBandwidth_kHz for image in images[::2]])
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
        # signal_ref = analysis.get_typical_level(images[0], mask_signal, mask_implant)

        slc = (slice(40, 160), slice(65, 185), slice(15, 45))
        images = images[(slice(None),) + slc]
        mask_empty = mask_empty[slc]
        mask_implant = mask_implant[slc]
        mask_signal = mask_signal[slc]

        num_trials = len(images) // 2
        snrs = []
        signals = []
        noise_stds = []

        # compute SNR
        if args.verbose:
            print('Computing SNR...')
        for i in range(num_trials):
            print('trial', i)
            image1 = images[2*i]
            image2 = images[2*i+1]
            snr, signal, noise_std = analysis.signal_to_noise(image1, image2, mask_signal, mask_empty)
            # noise_std = analysis.noise_std(image1, image2)
            snrs.append(snr)
            signals.append(signal)
            noise_stds.append(noise_std)

        # save outputs
        if args.verbose:
            print('Saving outputs...')

        np.savez(path.join(save_dir, 'outputs.npz'),
            images=images,
            snrs=np.stack(snrs),
            signals=np.stack(signals),
            noise_stds=np.stack(noise_stds),
            mask_signal=mask_signal,
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

    num_trials = len(images) // 2

    for i in range(num_trials):

        image1 = images[2*i]
        image2 = images[2*i+1]

        # volumes = (image1, image2, 10 * noise_stds[i] + 0.5, signals[i], snrs[i] / 80)
        # titles = ('Image 1 of pair', 'Image 2 of pair', 'Noise St. Dev. (10x)', 'Signal Mean', 'SNR (0 to 80)')
        # volumes = (image1, image2, signals[i], snrs[i] / 200)
        # titles = ('Image 1 of pair', 'Image 2 of pair', 'Signal Mean', 'SNR (0 to 200)')
        # fig1, tracker1 = plotVolumes(volumes, 1, len(volumes), titles=titles, figsize=(16, 8))

        image_diff = 5 * (image2 - image1) + 0.5
        image_sum = 0.5 * (image2 + image1)
        volumes = (image1, image2, image_diff, image_sum)
        titles = ('Image 1', 'Image 2', 'Difference (5x)', 'Sum (0.5x)')
        fig2, tracker2 = plotVolumes(volumes, 1, len(volumes), titles=titles, figsize=(16, 8))
    
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    plt.subplots_adjust(bottom=0.2)
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    plt.subplots_adjust(bottom=0.2)
    # colors = ['black', 'red', 'blue']
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    styles = ['dotted', 'solid', 'dashed']
    loosely_dashed = (0, (5, 10))
    ax3.axline((0, 0), (1, 1), color='k', linestyle=loosely_dashed)
    ax4.axline((0, 0), (1, 1), color='k', linestyle=loosely_dashed)
    fs = 20
    # noise_stds *= 100
    for i in range(1, num_trials):
        expected_factor = np.sqrt(rbw[0] / rbw[i])
        expected_snr_rounded = np.round(expected_factor * snrs[0])
        sns.lineplot(x=expected_snr_rounded.ravel(), y=snrs[i].ravel(), ax=ax3, legend='brief', label='RBW={:.3g}kHz'.format(rbw[i]), color=colors[i-1], linestyle=styles[i-1])  # plots mean line and 95% confidence band
        ax4.scatter(expected_factor * snrs[0], snrs[i], c=colors[i-1], label='RBW={:.3g}kHz'.format(rbw[i]), s=0.01, marker='.')
        # expected_noise_rounded = np.round(noise_stds[0] / expected_factor)
        # sns.lineplot(x=expected_noise_rounded.ravel(), y=noise_stds[i].ravel(), ax=ax3, legend='brief', label='{:.3g}kHz'.format(rbw[i]), color=colors[i-1])  # plots mean line and 95% confidence band
        # ax4.scatter(noise_stds[0] / expected_factor, noise_stds[i], c=colors[i-1], label='RBW={:.3g}kHz'.format(rbw[i]), s=0.01, marker='.')
    for ax in (ax3, ax4):
        ax.set_xlim([10, 55])
        ax.set_ylim([10, 55])
        ax.set_xticks(range(10, 51, 10))
        ax.set_yticks(range(10, 51, 10))
        ax.set_xlabel('Expected SNR', fontsize=fs)
        ax.set_ylabel('Measured SNR', fontsize=fs)
        ax.tick_params(labelsize=fs*0.8)
        ax.legend(fontsize=fs*0.8)
        ax.grid()
    fig3.savefig(path.join(save_dir, 'validation_snr.png'), dpi=300)
    fig4.savefig(path.join(save_dir, 'snr_pixel_cloud.png'), dpi=300)

    plt.show()
