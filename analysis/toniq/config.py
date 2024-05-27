"""Functions for working with TONIQ config files.

"""
import yaml
from os import path

from toniq.data import ImageVolume
from toniq.dicom import load_series_from_path

def read_config(file: str) -> dict:
    """ Read config file into a dictionary. """
    with open(file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def parse_slice(config: dict) -> tuple[slice]:
    """ Parse the slice field of config. """
    return tuple(slice(start, stop) for start, stop in config['slice'])

def load_volume(config: dict, name) -> ImageVolume:
    """ Load one image volume by name. """
    return load_series_from_path(path.join(config['root'], config['dicom-series'][name]))

def load_all_volumes(config: dict) -> dict[ImageVolume]:
    """ Load all image volumes named in config. """
    images = {}
    for name in config['dicom-series']:
        images[name] = load_volume(config, name)
    return images
