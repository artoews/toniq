def parse_slice(config):
    return tuple(slice(start, stop) for start, stop in config['params']['slice'])