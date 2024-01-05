import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from os import path, makedirs

from plot import plotVolumes

# from plot_distortion import image_results, field_results, summary_results
from plot_distortion import plot_image_results, plot_field_results, plot_summary_results
from distortion import map_distortion, get_registration_masks, setup_nonrigid, get_true_field
from util import equalize, load_series

# TODO systematize this
# slc = (slice(35, 155), slice(65, 185), 30)
# slc = (slice(25, 175), slice(50, 200), slice(10, 60))
slc = (slice(35, 155), slice(65, 185), slice(15, 45))

p = argparse.ArgumentParser(description='Geometric distortion analysis of 2DFSE multi-slice image volumes with varying readout bandwidth.')
p.add_argument('root', type=str, help='path where outputs are saved')
p.add_argument('-e', '--exam_root', type=str, default=None, help='directory where exam data exists in subdirectories')
p.add_argument('-s', '--series_list', type=str, nargs='+', default=None, help='list of exam_root subdirectories to be analyzed, with the first serving as plastic reference, second as metal reference')
p.add_argument('-t', '--threshold', type=float, default=0.1, help='maximum intensity artifact error included in registration mask')

if __name__ == '__main__':

    args = p.parse_args()
    save_dir = path.join(args.root, 'distortion')
    if not path.exists(save_dir):
        makedirs(save_dir)
    
    if args.exam_root is not None and args.series_list is not None:

        with open(path.join(save_dir, 'args.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=4)

        images = [load_series(args.exam_root, series_name) for series_name in args.series_list]

        pbw = np.array([image.meta.pixelBandwidth_Hz for image in images[1:]])
        rbw = np.array([image.meta.readoutBandwidth_kHz for image in images[1:]])

        images = np.stack([image.data for image in images])

        images = equalize(images)

        masks_register = get_registration_masks(images, args.threshold)
        
        if slc is not None:
            images = images[(slice(None),) + slc]
            masks_register = [mask[slc] for mask in masks_register]

        # volumes = (images[0], images[1], images[2], masks_register[0], masks_register[1])
        # titles = ('plastic', 'fixed', 'moving', 'fixed mask', 'moving mask')
        # fig2, tracker2 = plotVolumes(volumes, titles=titles, figsize=(16, 8))
        # plt.show()
        # quit()
        
        # run registration
        print('Running registration...')
        results = []
        deformation_fields = []
        fixed_mask = masks_register[0]
        itk_parameters = setup_nonrigid()
        for i in range(2, len(images)):
            print('on trial {} with PBWs {:.1f} and {:.1f} Hz'.format(i-1, pbw[0], pbw[i-1]))
            fixed_image = images[1]
            moving_image = images[i]
            moving_mask = masks_register[i-1]
            _, result_masked, deformation_field = map_distortion(
                fixed_image,
                moving_image,
                fixed_mask=fixed_mask,
                moving_mask=moving_mask,
                itk_parameters=itk_parameters)
            results.append(result_masked)
            deformation_fields.append(deformation_field)
        results = np.stack(results)
        masks_register = np.stack(masks_register)
        deformation_fields = np.stack(deformation_fields)

        np.savez(path.join(save_dir, 'outputs.npz'),
                 images=images,
                 results=results,
                 masks_register=masks_register,
                 deformation_fields=deformation_fields,
                 pbw=pbw,
                 rbw=rbw
                 )
        
    else:
        with open(path.join(save_dir, 'args_post.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=4)
        data = np.load(path.join(save_dir, 'outputs.npz'))
        for var in data:
            globals()[var] = data[var]

    true_field = get_true_field(path.join(args.root, 'field'))[slc]  # kHz
    
    if results.ndim == 3:
        slc = (slice(None), slice(None), slice(None))
    elif results.ndim == 4:
        slc = (slice(None), slice(None), slice(None), 15)
    print(masks_register.shape, images.shape, results.shape, true_field.shape, deformation_fields.shape)
    plot_image_results(plt.figure(figsize=(14, 5)), masks_register[slc], images[slc], results[slc], rbw)
    plt.savefig(path.join(save_dir, 'images.png'), dpi=300)
    plot_field_results(plt.figure(figsize=(8, 5)), results[slc], true_field[slc[1:]], deformation_fields[slc], rbw, pbw)
    plt.savefig(path.join(save_dir, 'fields.png'), dpi=300)
    plot_summary_results(plt.figure(), results[slc], true_field[slc[1:]], deformation_fields[slc + (slice(None),)], rbw, pbw) # TODO plot for entire 3D volume, not just the slice
    plt.savefig(path.join(save_dir, 'summary.png'), dpi=300)
    plt.show()
