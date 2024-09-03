import os
import re
from pathlib import Path
import glob
import random
import yaml
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def find_latest_ckpt(ckpt_dir, model_name):
    search_pattern = os.path.join(ckpt_dir, f"{model_name}_*_*.pt")
    ckpt_files = glob.glob(search_pattern)

    if not ckpt_files:
        return None

    pattern = re.compile(rf"{model_name}_(\d+)_(\d+)\.pt")

    # parse filenames and sort based on epoch and iteration
    def extract_epoch_iteration(file_path):
        match = pattern.search(os.path.basename(file_path))
        if match:
            epoch = int(match.group(1))
            iteration = int(match.group(2))
            return (epoch, iteration)
        else:
            return (0, 0)

    # Sort and get the latest files
    latest_ckpt = max(ckpt_files, key=extract_epoch_iteration)

    return latest_ckpt


def load_yaml_and_expand_var(file_path):
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)

    yaml_content = str(config)

    variables_in_yaml = re.findall(r'\$\{(\w+)\}', yaml_content)

    for var in set(variables_in_yaml):
        if var not in config:
            raise KeyError(f"Key '{var}' not found in the YAML file.")
        yaml_content = yaml_content.replace(f'${{{var}}}', config[var])

    expanded_config = yaml.safe_load(yaml_content)
    return expanded_config
