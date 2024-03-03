import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from os import path, makedirs
import scipy.ndimage as ndi
from skimage import morphology

from plot import plotVolumes
from plot_distortion import plot_image_results, plot_field_results, plot_summary_results
from distortion import get_distortion_map, get_registration_masks, setup_nonrigid
from util import equalize, load_series, save_args

from plot_params import *
from slice_params import *

slc = LATTICE_SLC

p = argparse.ArgumentParser(description='Geometric distortion analysis of 2DFSE multi-slice image volumes')
p.add_argument('root', type=str, help='path where outputs are saved')
p.add_argument('-e', '--exam_root', type=str, default=None, help='directory where exam data exists in subdirectories')
p.add_argument('-s', '--series_list', type=str, nargs='+', default=None, help='list of exam_root subdirectories to be analyzed in RBW-matched (plastic, metal) pairs')
p.add_argument('-t', '--threshold', type=float, default=0.3, help='maximum intensity artifact error included in registration mask; default=0.3')
p.add_argument('-p', '--plot', action='store_true', help='show plots')

if __name__ == '__main__':

    args = p.parse_args()
    save_dir = path.join(args.root, 'distortion')
    ia_dir = path.join(args.root, 'artifact')
    if not path.exists(save_dir):
        makedirs(save_dir)
    
    if args.exam_root is not None and args.series_list is not None:

        save_args(args, save_dir)

        images = [load_series(args.exam_root, series_name) for series_name in args.series_list]

        num_trials = len(images) // 2
        pbw = np.array([image.meta.pixelBandwidth_Hz for image in images[::2]])
        rbw = np.array([image.meta.readoutBandwidth_kHz for image in images[::2]])

        images = np.stack([image.data for image in images])
        images = equalize(images)
        images = images[(slice(None),) + slc]

        implant_mask = np.load(path.join(ia_dir, 'implant-mask.npy'))
        ia_maps = np.load(path.join(ia_dir, 'ia-maps.npy'))
        # ia_maps = [ndi.generic_filter(artifact_map, np.mean, footprint=morphology.cube(5)) for artifact_map in ia_maps]
        mask_pairs = [get_registration_masks(implant_mask, ia_maps[i], args.threshold) for i in range(num_trials)]

        # fig, tracker = plotVolumes(mask_pairs[0])
        # plt.show()
        # quit()

        # run registration
        print('Running registration...')
        results = []
        deformation_fields = []
        itk_parameters = setup_nonrigid()
        for i in range(len(images) // 2):
            print('on trial {} of {}'.format(i+1, num_trials))
            fixed_image = images[2*i].copy()
            moving_image = images[2*i+1].copy()
            fixed_mask, moving_mask = mask_pairs[i]
            result, result_masked, deformation_field = get_distortion_map(fixed_image, moving_image, fixed_mask, moving_mask, itk_parameters=itk_parameters)
            results.append(result_masked)
            deformation_fields.append(deformation_field)
        results = np.stack(results)
        masks = np.stack(sum(mask_pairs, ())) # consolidate list of tuples
        deformation_fields = np.stack(deformation_fields)

        np.savez(path.join(save_dir, 'outputs.npz'),
                 images=images,
                 results=results,
                 masks=masks,
                 deformation_fields=deformation_fields,
                 pbw=pbw,
                 rbw=rbw
                 )
        
        np.save(path.join(save_dir, 'images.npy'), images)
        np.save(path.join(save_dir, 'gd-maps.npy'), deformation_fields)
        np.save(path.join(save_dir, 'gd-masks.npy'), masks)
        
    else:
        with open(path.join(save_dir, 'args_post.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=4)
        data = np.load(path.join(save_dir, 'outputs.npz'))
        for var in data:
            globals()[var] = data[var]

    true_field_kHz = np.load(path.join(args.root, 'field', 'field_diff_Hz.npy'))[slc] / 1000
    
    if results.ndim == 3:
        slc = (slice(None), slice(None), slice(None))
    elif results.ndim == 4:
        slc_z = (slice(None), slice(None), slice(None), images.shape[-1]//2)
        slc_y = (slice(None), slice(None), images.shape[-2]//2, slice(None))

    plot_image_results(plt.figure(figsize=(12, 5)), masks, images, results)
    plt.savefig(path.join(save_dir, 'images.png'), dpi=300)

    # plot_field_results(plt.figure(figsize=(12, 5)), results, true_field_kHz, deformation_fields, rbw, pbw, field_dir=0)
    # plt.savefig(path.join(save_dir, 'fields_x.png'), dpi=300)
    # plot_field_results(plt.figure(figsize=(8, 3)), results, true_field_kHz, deformation_fields, rbw, pbw, field_dir=1)
    # plt.savefig(path.join(save_dir, 'fields_y.png'), dpi=300)
    # plot_field_results(plt.figure(figsize=(12, 5)), results, true_field_kHz, deformation_fields, rbw, pbw, field_dir=2)
    # plt.savefig(path.join(save_dir, 'fields_z.png'), dpi=300)

    # plot_summary_results(plt.figure(), results, true_field_kHz, -deformation_fields[..., 0], rbw, pbw)
    # plt.savefig(path.join(save_dir, 'summary_x.png'), dpi=300)
    # plot_summary_results(plt.figure(), results, 0 * true_field_kHz, deformation_fields[..., 1], rbw, pbw * np.inf)
    # plt.savefig(path.join(save_dir, 'summary_y.png'), dpi=300)
    kHz_mm_over_G_cm = 0.42577
    # plot_summary_results(plt.figure(), results, true_field_kHz, deformation_fields[..., 2], rbw, (1.499 * kHz_mm_over_G_cm * 1.2 * 1000,) * len(pbw))
    # plt.savefig(path.join(save_dir, 'summary_z.png'), dpi=300)

    # volumes = (true_field_kHz, -deformation_fields[0][..., 0], deformation_fields[0][..., 2])
    # titles = ('correct field', 'deformation x', 'deformation z')
    # fig1, tracker1 = plotVolumes(volumes, titles=titles, vmin=-4, vmax=4, cmap='RdBu_r')
    # fig2, tracker2 = plotVolumes((images[1], images[2], results[0], images[1]-images[1], np.abs(images[2]-images[1]), np.abs(results[0]-images[1])))

    # volumes = (true_field_kHz, -deformation_fields[0][..., 0], deformation_fields[0][..., 2])
    # titles = ('correct field', 'deformation x', 'deformation z')
    # fig1, tracker1 = plotVolumes(volumes, titles=titles, vmin=-4, vmax=4, cmap='RdBu_r')
    # fig2, tracker2 = plotVolumes((images[1], images[2], results[0], images[1]-images[1], np.abs(images[2]-images[1]), np.abs(results[0]-images[1])))

    if args.plot:
        plt.show()