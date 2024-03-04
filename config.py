import yaml

def read_config(file):
    with open(file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def parse_slice(config):
    return tuple(slice(start, stop) for start, stop in config['params']['slice'])