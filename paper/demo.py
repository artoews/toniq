import argparse
import matplotlib.pyplot as plt
import numpy as np
import sigpy as sp
import yaml
from pathlib import Path

from os import path, makedirs
from skimage import transform

from artifact import get_artifact_map
from plot_artifact import plot_artifact_results
from distortion import get_registration_masks, get_distortion_map
from plot_distortion import plot_image_results
from plot import plotVolumes
from plot_params import *
from resolution import map_resolution
from intensity import map_snr
from masks import get_implant_mask, get_signal_mask, get_artifact_mask
from util import equalize, load_series_from_path, masked_copy

p = argparse.ArgumentParser(description='Run all four mapping analyses on a single sequence')
p.add_argument('root', type=str, help='path where outputs are saved')
p.add_argument('config', type=str, default=None, help='yaml config file specifying data paths and mapping parameters')
p.add_argument('--ia', action='store_true', help='do intensity artifact map')
p.add_argument('--gd', action='store_true', help='do geometric distortion map')
p.add_argument('--snr', action='store_true', help='do SNR map')
p.add_argument('--res', action='store_true', help='do resolution map')

def parse_slice(config):
    return tuple(slice(start, stop) for start, stop in config['params']['slice'])

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
    images = {}
    for name, series_path in config['dicom-series'].items():
        images[name] = load_series_from_path(series_path)
    
    # IA mapping
    if args.ia or map_all:
        plastic_image, metal_image = prepare_inputs((images['uniform-plastic'].data, images['uniform-metal'].data), slc)
        implant_mask = get_implant_mask(plastic_image)
        ia_map = get_artifact_map(plastic_image, metal_image, implant_mask)
        np.save(path.join(save_dir, 'ia-plastic.npy'), plastic_image)
        np.save(path.join(save_dir, 'ia-metal.npy'), metal_image)
        np.save(path.join(save_dir, 'ia-map.npy'), ia_map)
        np.save(path.join(save_dir, 'implant-mask.npy'), implant_mask)
        plot_artifact_results((plastic_image, metal_image), (ia_map,))
        fig0, tracker0 = plotVolumes((plastic_image, metal_image))
        fig1, tracker1 = plotVolumes((ia_map,), cmap=CMAP['artifact'], vmin=-0.6, vmax=0.6, cbar=True)

    # GD mapping
    if args.gd or map_all:
        ia_map = np.load(path.join(save_dir, 'ia-map.npy'))
        implant_mask = np.load(path.join(save_dir, 'implant-mask.npy'))
        plastic_image, metal_image = prepare_inputs((images['structured-plastic'].data, images['structured-metal'].data), slc)
        plastic_mask, metal_mask = get_registration_masks(implant_mask, ia_map, config['params']['IA-thresh-relative'])
        result, result_masked, rigid_result, rigid_result_masked, gd_map = get_distortion_map(plastic_image, metal_image, plastic_mask, metal_mask)
        np.save(path.join(save_dir, 'gd-plastic.npy'), plastic_image)
        np.save(path.join(save_dir, 'gd-plastic-mask.npy'), plastic_mask)
        np.save(path.join(save_dir, 'gd-metal.npy'), metal_image)
        np.save(path.join(save_dir, 'gd-metal-mask.npy'), metal_mask)
        np.save(path.join(save_dir, 'gd-metal-registered.npy'), result)
        np.save(path.join(save_dir, 'gd-metal-registered-masked.npy'), result_masked)
        np.save(path.join(save_dir, 'gd-metal-rigid-registered.npy'), rigid_result)
        np.save(path.join(save_dir, 'gd-metal-rigid-registered-masked.npy'), rigid_result_masked)
        np.save(path.join(save_dir, 'gd-map.npy'), gd_map)
        plot_image_results(plt.figure(), (plastic_mask, metal_mask), (plastic_image, metal_image), (result_masked,))
        plastic_image_masked = masked_copy(plastic_image, plastic_mask)
        metal_image_masked = masked_copy(metal_image, metal_mask)
        # fig2, tracker2 = plotVolumes((plastic_image, metal_image, result, plastic_mask, metal_mask, result_masked), nrows=2, ncols=3)
        fig2, tracker2 = plotVolumes((plastic_image_masked, metal_image_masked, result_masked))
        fig3, tracker3 = plotVolumes((-gd_map[..., 0], gd_map[..., 1], gd_map[..., 2]), cmap=CMAP['distortion'], vmin=-2, vmax=2, cbar=True)

    # SNR mapping
    if args.snr or map_all:
        ia_map = np.load(path.join(save_dir, 'ia-map.npy'))
        implant_mask = np.load(path.join(save_dir, 'implant-mask.npy'))
        image_1, image_2 = prepare_inputs((images['uniform-metal'].data, images['uniform-metal-2'].data), slc)
        # image_1, image_2 = prepare_inputs((images['uniform-plastic'].data, images['uniform-plastic-2'].data), slc)
        ia_mask = get_artifact_mask(ia_map, config['params']['IA-thresh-relative'])
        snr_mask = get_signal_mask(implant_mask, artifact_masks=[ia_mask])
        # snr_mask = get_signal_mask(implant_mask) # good for evaluation on plastic
        snr, signal, noise_std = map_snr(image_1, image_2, snr_mask)
        np.save(path.join(save_dir, 'snr-image-1.npy'), image_1)
        np.save(path.join(save_dir, 'snr-image-2.npy'), image_2)
        np.save(path.join(save_dir, 'snr-mask.npy'), snr_mask)
        np.save(path.join(save_dir, 'snr-map.npy'), snr)
        fig4, tracker4 = plotVolumes((image_1, image_2, snr/200))

    # Resolution mapping
    if args.res or map_all:
        gd_map = np.load(path.join(save_dir, 'gd-map.npy'))
        ia_map = np.load(path.join(save_dir, 'ia-map.npy'))
        ia_mask = get_artifact_mask(ia_map, config['params']['IA-thresh-relative'])
        implant_mask = np.load(path.join(save_dir, 'implant-mask.npy'))
        image_ref = images['structured-plastic-reference'].data
        # image_blurred = images['structured-metal'].data
        image_blurred = images['structured-plastic'].data
        image_ref = np.abs(sp.ifft(sp.resize(sp.fft(image_ref), image_blurred.shape)))
        # image_blurred = np.abs(sp.ifft(sp.resize(sp.fft(image_blurred), image_ref.shape)))
        # slc = tuple(slice(s.start*2, s.stop*2) for s in slc[:2]) + slc[2:]
        image_ref, image_blurred = prepare_inputs((image_ref, image_blurred), slc)
        resolution_mm = images['structured-plastic'].meta.resolution_mm
        patch_shape = tuple(config['params']['psf-window-size'])
        num_workers = config['params']['num-workers']
        gd_masks = [get_artifact_mask(gd_map[..., i], config['params']['GD-thresh-pixels']) for i in range(3)]
        # mask = get_signal_mask(implant_mask, artifact_masks=[ia_mask] + gd_masks)
        # mask = get_signal_mask(implant_mask, artifact_masks=[ia_mask]) # ignores GD mask; good for MSL protocols
        mask = get_signal_mask(implant_mask) # good for evaluation on plastic
        mask = transform.resize(mask, image_ref.shape)
        psf, fwhm = map_resolution(image_ref, image_blurred, patch_shape, resolution_mm, mask, config['params']['psf-stride'], num_workers=num_workers)
        np.save(path.join(save_dir, 'res-image-ref.npy'), image_ref)
        np.save(path.join(save_dir, 'res-image-blurred.npy'), image_blurred)
        np.save(path.join(save_dir, 'res-mask.npy'), mask)
        np.save(path.join(save_dir, 'psf-map.npy'), psf)
        np.save(path.join(save_dir, 'fwhm-map.npy'), fwhm)
        fig5, tracker5 = plotVolumes((image_ref, image_blurred, mask))
        res_x_map = fwhm[..., 0] / resolution_mm[0]
        res_y_map = fwhm[..., 1] / resolution_mm[1]
        fig6, tracker6 = plotVolumes((res_x_map, res_y_map), vmin=1, vmax=3, cmap=CMAP['resolution'], cbar=True)

    # just for debugging as I write the script
    plt.show()
