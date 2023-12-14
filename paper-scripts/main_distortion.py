import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from os import path, makedirs
import scipy.ndimage as ndi
from skimage import morphology
from plot import plotVolumes
import register

from plot_distortion import image_results, field_results, summary_results
from register import map_distortion, get_registration_masks
from util import equalize, load_series

# TODO systematize this
slc = (slice(35, 155), slice(65, 185), 30)
# slc = (slice(25, 175), slice(50, 200), slice(10, 60))
# slc = (slice(35, 155), slice(65, 185), slice(15, 45))

p = argparse.ArgumentParser(description='Geometric distortion analysis of 2DFSE multi-slice image volumes with varying readout bandwidth.')
p.add_argument('root', type=str, help='path where outputs are saved')
p.add_argument('-e', '--exam_root', type=str, default=None, help='directory where exam data exists in subdirectories')
p.add_argument('-s', '--series_list', type=str, nargs='+', default=None, help='list of exam_root subdirectories to be analyzed, with the first serving as plastic reference, second as metal reference')

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

        masks_register = get_registration_masks(images)
        
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
        itk_parameters = register.setup_nonrigid()
        for i in range(2, len(images)):
            print('on trial {} with PBWs {:.1f} and {:.1f} Hz'.format(i, pbw[0], pbw[i-1]))
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
    
    # true_field = np.load(path.join(args.root, 'field', 'field.npy'))  # kHz
    # true_field = np.load(path.join(args.root, 'field-metal', 'field.npy')) - np.load(path.join(args.root, 'field-plastic', 'field.npy'))  # kHz
    metal_field = np.load(path.join(args.root, 'field', 'field-metal.npy'))
    plastic_field = np.load(path.join(args.root, 'field', 'field-plastic.npy'))  # kHz
    true_field = metal_field - plastic_field
    true_field = ndi.median_filter(true_field, footprint=morphology.ball(4))
    # true_field = ndi.generic_filter(true_field, np.mean, footprint=morphology.ball(3))
    # true_field = true_field[slc][slc2[1:]] * 1000  # Hz
    true_field = true_field[slc] # kHz

    # fig4, tracker4 = plotVolumes((images[0] * 24e3 - 12e3, true_field), titles=('trial 0', 'true_field'), vmin=-12e3, vmax=12e3)
    # true_field_masked = masked_copy(true_field, fixed_mask)

    fig1, axes1 = image_results(images, masks_register, results, rbw, save_dir=save_dir)
    fig2, axes2 = field_results(true_field, deformation_fields, results, rbw, pbw, save_dir=save_dir)
    fig3, axes3 = summary_results(true_field, deformation_fields, results, rbw, pbw, save_dir=save_dir)
    plt.show()
