# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import plot_tensor, save_plot
import yaml
import argparse
import random
import pathlib
import wandb

# import models
from gradtts import GradTTS

from dataset import Dataset, collateFn
import set_seed

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="path to config file")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    print("Initializing data loaders...")
    with open(args.config) as f:
        config = yaml.load(f, yaml.SafeLoader)
    log_dir = pathlib.Path(config["log_dir"])

    # set seed
    set_seed.set_seed(config["random_seed"])

    print('Initializing data loaders...')
    train_dataset = Dataset(
        config["train_datalist_path"],
        config["phn2id_path"],
        config["sample_rate"],
        config["n_fft"],
        config["n_mels"],
        config["f_min"],
        config["f_max"],
        config["hop_size"],
        config["win_size"],
    )

    val_dataset = Dataset(
        config["valid_datalist_path"],
        config["phn2id_path"],
        config["sample_rate"],
        config["n_fft"],
        config["n_mels"],
        config["f_min"],
        config["f_max"],
        config["hop_size"],
        config["win_size"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        collate_fn=collateFn,
        num_workers=16,
    )

    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=collateFn
    )


