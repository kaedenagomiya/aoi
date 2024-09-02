import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#import torchvision.transforms as transforms

gpu_name = torch.cuda.get_device_name()
if gpu_name is None:
    gpu_name = "None"

import toybox
import wandb
from wandb import AlertLevel
import random

# seq define params
RANDOM_SEED=42

# seq train
toybox.set_seed(RANDOM_SEED)

total_runs = 5

config_wandb={
"learning_rate": 0.02,
"architecture": "CNN",
"dataset": "CIFAR-100",
"optimizer": "Adam",
"criterion": "loss_mse",
"epochs": 10,
}

for run in range(total_runs):

    # wandb init
    #wandb.init(project='mandara_test',group=f'run_{run}',
    #        name='test_run',config=config_wandb)
    # 1. ---------------------------------------------------------------------------
    wandb.init(project='aoi', name='aoi_test_run', config=config_wandb)

    # train script
    epochs = 10
    offset = random.random() / 5
    for epoch in range(2, epochs):
        accuracy = 1 - 2 ** -epoch - random.random() / epoch - offset
        loss = 2 ** -epoch + random.random() / epoch + offset
        # record
        # 2. ---------------------------------------------------------------------------
        wandb.log({"accuracy": accuracy, "loss": loss})


    # wandb alert
    if run >= 4:
        # 3. ---------------------------------------------------------------------------
        wandb.alert(
            title='fin aoi_test_run each epoch',
            text=f'<@U052F2QKFMK> GPU:${gpu_name} learning now (> w <), acc: {accuracy} ',
            level=AlertLevel.INFO
        )


# end
# 4. ---------------------------------------------------------------------------
wandb.finish()
