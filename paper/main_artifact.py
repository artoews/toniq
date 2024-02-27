import argparse
import matplotlib.pyplot as plt
import numpy as np
from os import path, makedirs

from artifact import get_artifact_map, get_signal_reference
from masks import get_implant_mask
from plot_artifact import plot_artifact_results, plot_signal_ref
from plot import plotVolumes, imshow2

from util import equalize, load_series, save_args, masked_copy
from slice_params import *
from plot_params import *

slc = LATTICE_SLC

p = argparse.ArgumentParser(description='Intensity artifact analysis of 2DFSE multi-slice image volumes with varying readout bandwidth.')
p.add_argument('root', type=str, help='path where outputs are saved')
p.add_argument('-e', '--exam_root', type=str, default=None, help='directory where exam data exists in subdirectories')
p.add_argument('-s', '--series_list', type=str, nargs='+', default=None, help='list of exam_root subdirectories to be analyzed in RBW-matched (plastic, metal) pairs')
p.add_argument('-p', '--plot', action='store_true', help='show plots')

if __name__ == '__main__':

    args = p.parse_args()
    save_dir = path.join(args.root, 'artifact')
    images_file = path.join(save_dir, 'images.npy')
    maps_file = path.join(save_dir, 'ia-maps.npy')
    mask_file = path.join(save_dir, 'implant-mask.npy')
    sigref_file = path.join(save_dir, 'signal-reference.npy')
    if not path.exists(save_dir):
        makedirs(save_dir)

    if args.exam_root is not None and args.series_list is not None:

        save_args(args, save_dir)

        images = [load_series(args.exam_root, series_name) for series_name in args.series_list]
        num_trials = len(images) // 2
        rbw = np.array([image.meta.readoutBandwidth_kHz for image in images[::2]])

        images = np.stack([image.data for image in images])
        images = equalize(images)
        images = images[(slice(None),) + slc]

        implant_mask = get_implant_mask(images[0])
        ia_maps = [get_artifact_map(images[2*i], images[2*i+1], implant_mask) for i in range(num_trials)]
        sig_refs = [get_signal_reference(images[2*i], implant_mask) for i in range(num_trials)]
        ia_maps = np.stack(ia_maps)
        sig_refs = np.stack(sig_refs)

        np.save(images_file, images)
        np.save(maps_file, ia_maps)
        np.save(mask_file, implant_mask)
        np.save(sigref_file, sig_refs)
    
    else:
        images = np.load(images_file)
        ia_maps = np.load(maps_file)
        implant_mask = np.load(mask_file)
        sig_refs = np.load(sigref_file)
    
    plot_artifact_results(images, ia_maps, save_dir=save_dir)
    plot_signal_ref(images, sig_refs, save_dir=save_dir)
    # image_masked = masked_copy(images[0], implant_mask)
    # fig, tracker = plotVolumes((images[0], image_masked, sig_refs[0]), nrows=1, ncols=3) # for debugging

    if args.plot:
        plt.show()