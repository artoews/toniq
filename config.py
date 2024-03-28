import yaml
from os import path
from util import load_series_from_path

def read_config(file):
    with open(file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def parse_slice(config):
    return tuple(slice(start, stop) for start, stop in config['slice'])

def load_volume(config, name):
    return load_series_from_path(path.join(config['root'], config['dicom-series'][name]))

def load_all_volumes(config):
    images = {}
    for name in config['dicom-series']:
        images[name] = load_volume(config, name)
    return images
