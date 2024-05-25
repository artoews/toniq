import argparse
import matplotlib.pyplot as plt
import numpy as np
import sigpy as sp
import yaml
from pathlib import Path

from os import path, makedirs

from toniq import ia, snr, gd, sr
from toniq.config import parse_slice, load_all_volumes
from toniq.plot import plotVolumes
from toniq.plot_params import *
from toniq.masks import get_implant_mask, get_signal_mask, get_artifact_mask
from toniq.util import equalize, masked_copy, safe_divide

p = argparse.ArgumentParser(description='Run all four mapping analyses on a single sequence')
p.add_argument('config', type=str, default=None, help='data config file')
p.add_argument('root', type=str, help='folder where save directory is created, i.e. parent folder of save_dir')
p.add_argument('--ia', action='store_true', help='do intensity artifact map')
p.add_argument('--gd', action='store_true', help='do geometric distortion map')
p.add_argument('--snr', action='store_true', help='do SNR map')
p.add_argument('--res', action='store_true', help='do resolution map')
p.add_argument('--plastic', action='store_true', help='use only plastic inputs where possible (snr, res)')
p.add_argument('--ia_thresh', type=float, default=0.4, help='threshold for IA mask, with relative units; default=0.4')
p.add_argument('--gd_thresh', type=float, default=1, help='threshold for GD mask, with units pixels; default=1')
p.add_argument('--psf_window_size', type=int, nargs=3, default=[14, 14, 10], help='size of window used for SR mapping; default=[14, 14, 10]')
p.add_argument('--psf_shape', type=int, nargs=3, default=[5, 5, 1], help='size of PSF used for SR mapping; default=[5, 5, 1]')
p.add_argument('--psf_stride', type=int, default=1, help='stride used for SR mapping; default=1')
p.add_argument('--num_workers', type=int, default=8, help='number of workers used for SR mapping; default=8')
p.add_argument('-p', '--plot', action='store_true', help='show plots')


def prepare_inputs(images, slc):
    images = equalize(images)
    images = [image[slc] for image in images]
    return images


if __name__ == '__main__':

    # process args
    args = p.parse_args()
    save_dir = path.join(args.root, Path(args.config).stem)
    if not path.exists(save_dir):
        makedirs(save_dir)

    if args.ia or args.gd or args.snr or args.res:
        map_all = False
    else:
        map_all = True

    # process config
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    with open(path.join(save_dir, 'config.yml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    slc = parse_slice(config)

    # load image data
    images = load_all_volumes(config)
    
    # IA mapping
    if args.ia or map_all:
        plastic_image, metal_image = prepare_inputs((images['uniform-plastic'].data, images['uniform-metal'].data), slc)
        implant_mask = get_implant_mask(plastic_image)
        ia_map = ia.get_map(plastic_image, metal_image, implant_mask)
        np.save(path.join(save_dir, 'ia-plastic.npy'), plastic_image)
        np.save(path.join(save_dir, 'ia-metal.npy'), metal_image)
        np.save(path.join(save_dir, 'ia-map.npy'), ia_map)
        np.save(path.join(save_dir, 'implant-mask.npy'), implant_mask)
        fig0, tracker0 = plotVolumes((plastic_image, metal_image), titles=('Plastic', 'Metal'))
        fig0.suptitle('Intensity Artifact Mapping Inputs')
        fig1, tracker1 = plotVolumes((ia_map,), titles=('IA Map',), cmap=CMAP['artifact'], vmin=-0.8, vmax=0.8, cbar=True)
        fig1.suptitle('Intensity Artifact Map')

    # GD mapping
    if args.gd or map_all:
        ia_plastic = np.load(path.join(save_dir, 'ia-plastic.npy'))
        ia_metal = np.load(path.join(save_dir, 'ia-metal.npy'))
        ia_map = np.load(path.join(save_dir, 'ia-map.npy'))
        implant_mask = np.load(path.join(save_dir, 'implant-mask.npy'))
        plastic_image, metal_image = prepare_inputs((images['structured-plastic'].data, images['structured-metal'].data), slc)
        plastic_mask, metal_mask = gd.get_masks(implant_mask, ia_map, args.ia_thresh)
        result, result_masked, rigid_result, rigid_result_masked, gd_map, rigid_transform, nonrigid_transform = gd.get_map(plastic_image, metal_image, plastic_mask, metal_mask)
        np.save(path.join(save_dir, 'gd-plastic.npy'), plastic_image)
        np.save(path.join(save_dir, 'gd-plastic-mask.npy'), plastic_mask)
        np.save(path.join(save_dir, 'gd-metal.npy'), metal_image)
        np.save(path.join(save_dir, 'gd-metal-mask.npy'), metal_mask)
        np.save(path.join(save_dir, 'gd-plastic-registered.npy'), result)
        np.save(path.join(save_dir, 'gd-plastic-registered-masked.npy'), result_masked)
        # np.save(path.join(save_dir, 'gd-plastic-rigid-registered.npy'), rigid_result)
        # np.save(path.join(save_dir, 'gd-plastic-rigid-registered-masked.npy'), rigid_result_masked)
        np.save(path.join(save_dir, 'gd-map.npy'), gd_map)
        plastic_image_masked = masked_copy(plastic_image, plastic_mask)
        metal_image_masked = masked_copy(metal_image, metal_mask)
        input_mask = (plastic_image_masked != 0) * (metal_image_masked != 0)
        output_mask = (result_masked != 0) * (metal_image_masked != 0)
        np.save(path.join(save_dir, 'gd-map-mask.npy'), output_mask)
        input_error = np.abs(plastic_image_masked - metal_image_masked) * input_mask
        output_error = np.abs(result_masked - metal_image_masked) * output_mask
        fig2, tracker2 = plotVolumes((metal_image_masked, plastic_image_masked, result_masked, input_error, output_error),
                                     titles=('Metal Input', 'Plastic Input', 'Plastic Output', 'Input Error', 'Output Error'))
        fig2.suptitle('Geometric Distortion, Mapping Images')
        fig3, tracker3 = plotVolumes((-gd_map[..., 0] * output_mask, gd_map[..., 1] * output_mask, gd_map[..., 2] * output_mask, (safe_divide(-gd_map[..., 0], gd_map[...,  2])-1) * output_mask),
                                     titles=('X (pixels)', 'Y (pixels)', 'Z (pixels)', '(X / Z) - 1'),
                                     cmap=CMAP['distortion'], vmin=-2, vmax=2, cbar=True)
        fig3.suptitle('Geometric Distortion Maps (Masked)')
        print('max GD in Y (pixels)', np.max(gd_map[..., 1]))

    # SNR mapping
    if args.snr or map_all:
        ia_map = np.load(path.join(save_dir, 'ia-map.npy'))
        implant_mask = np.load(path.join(save_dir, 'implant-mask.npy'))
        ia_mask = get_artifact_mask(ia_map, args.ia_thresh)
        if args.plastic:
            image_1, image_2 = prepare_inputs((images['uniform-plastic'].data, images['uniform-plastic-2'].data), slc)
            snr_mask = get_signal_mask(implant_mask)
        else:
            image_1, image_2 = prepare_inputs((images['uniform-metal'].data, images['uniform-metal-2'].data), slc)
            snr_mask = get_signal_mask(implant_mask, artifact_masks=[ia_mask])
        snr_map = snr.get_map(image_1, image_2, snr_mask)
        np.save(path.join(save_dir, 'snr-image-1.npy'), image_1)
        np.save(path.join(save_dir, 'snr-image-2.npy'), image_2)
        np.save(path.join(save_dir, 'snr-mask.npy'), snr_mask)
        np.save(path.join(save_dir, 'snr-map.npy'), snr_map)
        fig4, tracker4 = plotVolumes((image_1, image_2, snr_map/200), titles=('Image 1', 'Image 2', 'SNR Map (1/200)'))
        fig4.suptitle('SNR Mapping Inputs and Output')

    # Resolution mapping
    if args.res or map_all:
        gd_map = np.load(path.join(save_dir, 'gd-map.npy'))
        ia_map = np.load(path.join(save_dir, 'ia-map.npy'))
        ia_mask = get_artifact_mask(ia_map, args.ia_thresh)
        implant_mask = np.load(path.join(save_dir, 'implant-mask.npy'))
        image_ref = images['structured-plastic-reference'].data
        if args.plastic:
            image_blurred = images['structured-plastic'].data
            mask = get_signal_mask(implant_mask)
        else:
            image_blurred = images['structured-metal'].data
            gd_masks = [get_artifact_mask(gd_map[..., i], args.gd_thresh) for i in range(3)]
            # mask = get_signal_mask(implant_mask)
            mask = get_signal_mask(implant_mask, artifact_masks=[ia_mask] + gd_masks)
            # mask = get_signal_mask(implant_mask, artifact_masks=[ia_mask]) # ignores GD mask; good for MSL protocols
        image_ref = np.abs(sp.ifft(sp.resize(sp.fft(image_ref), image_blurred.shape)))
        image_ref, image_blurred = prepare_inputs((image_ref, image_blurred), slc)
        resolution_mm = images['structured-plastic'].meta.resolution_mm
        psf, fwhm = sr.get_map(image_ref, image_blurred, tuple(args.psf_shape), tuple(args.psf_window_size), resolution_mm, mask, args.psf_stride, num_workers=args.num_workers)
        np.save(path.join(save_dir, 'res-image-ref.npy'), image_ref)
        np.save(path.join(save_dir, 'res-image-blurred.npy'), image_blurred)
        np.save(path.join(save_dir, 'res-mask.npy'), mask)
        np.save(path.join(save_dir, 'psf-map.npy'), psf)
        np.save(path.join(save_dir, 'fwhm-map.npy'), fwhm)
        fig5, tracker5 = plotVolumes((image_ref, image_blurred, mask), titles=('Reference', 'Target', 'Mask'))
        fig5.suptitle('Spatial Resolution Mapping Inputs')
        res_x_map = fwhm[..., 0] / resolution_mm[0]
        res_y_map = fwhm[..., 1] / resolution_mm[1]
        fig6, tracker6 = plotVolumes((res_x_map, res_y_map), titles=('FWHM X (pixels)', 'FWHM Y (pixels)'), vmin=1, vmax=3, cmap=CMAP['resolution'], cbar=True)
        fig6.suptitle('Spatial Resolution Maps')

    if args.plot:
        plt.show()
