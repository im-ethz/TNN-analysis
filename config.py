import yaml

with open('mapping.yml', 'r') as f:
    rider_mapping = yaml.safe_load(f)

rider_mapping_inv = {v:k for k,v in rider_mapping.items()}

DATA_PATH = '/mnt/wave/hypex/data/'
SAVE_PATH = 'results/descriptives/'

OUTLET = 'ATTD'#'DC'