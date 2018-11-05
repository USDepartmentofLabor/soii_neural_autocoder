import warnings
import yaml

try:
    with open('config.yml', 'r') as f:
        config = yaml.load(f)
except FileNotFoundError:
    warnings.warn('no config.yml found, using default instead')
    config = {'GPU_HOST': ''}
