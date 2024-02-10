import argparse
import matplotlib.pyplot as plt
import numpy as np
import yaml

from os import path, makedirs

from artifact import get_artifact_map
from plot_artifact import plot_artifact_results
from masks import get_implant_mask
from util import equalize, load_series_from_path

p = argparse.ArgumentParser(description='Run all four mapping analyses on a single sequence')
p.add_argument('root', type=str, help='path where outputs are saved')
p.add_argument('config', type=str, default=None, help='yaml config file specifying data paths and mapping parameters')

def parse_slice(config):
    return tuple(slice(start, stop) for start, stop in config['params']['slice'])

def prepare_inputs(images, slc):
    images = equalize(images)
    images = [image[slc] for image in images]
    return images


if __name__ == '__main__':

    # process args
    args = p.parse_args()
    if not path.exists(args.root):
        makedirs(args.root)

    # process config
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    with open(path.join(args.root, 'config.yml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    slc = parse_slice(config)

    # load image data
    images = {}
    for name, series_path in config['dicom-series'].items():
        images[name] = load_series_from_path(series_path)
    
    # IA mapping
    plastic_image, metal_image = prepare_inputs((images['empty-plastic'].data, images['empty-metal'].data), slc)
    implant_mask = get_implant_mask(plastic_image)
    ia_map = get_artifact_map(plastic_image, metal_image, implant_mask)
    np.save(path.join(args.root, 'ia-plastic.npy'), plastic_image)
    np.save(path.join(args.root, 'ia-metal.npy'), metal_image)
    np.save(path.join(args.root, 'ia-map.npy'), ia_map)
    np.save(path.join(args.root, 'implant-mask.npy'), implant_mask)

    # TODO GD, SNR, resolution in the same manner as above
    # TODO write an accompanying plotting script to take a demo root folder and make the last figure of the paper

    # just for debugging as I write the script
    plot_artifact_results((plastic_image, metal_image), (ia_map,))
    plt.show()
    
