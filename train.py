# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import sys
import argparse
import random
import pathlib
import yaml
import numpy as np
from tqdm import tqdm


import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import Dataset, collateFn

from utils import plot_tensor, save_plot
import toybox

# import models
from gradtts import GradTTS
from gradseptts import GradSepTTS
from gradtfktts import GradTFKTTS
from gradtfk5tts import GradTFKTTS as GradTFK5TTS
from gradtimektts import GradTimeKTTS
from gradtfkfultts import GradTFKFULTTS

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="path to config file")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    
    # for init config =======================================================
    print("Initializing config...")
    #with open(args.config) as f:
    #    config = yaml.load(f, yaml.SafeLoader)
    try:
        print(args.config)
        print(str(type(args.config)))
        config = toybox.load_yaml_and_expand_var(args.config)
        print("log_dir:" + str(config["log_dir"]))
        print("ckpt_dir:" + str(config["ckpt"]))
    except KeyError as e:
        print(e)

    log_dir = pathlib.Path(config["log_dir"])
    model_name = config['model_name']
    if config["device"] == "GPU":
        assert torch.cuda.is_available(), "GPU environment is not specified."
        # if torch.cuda.is_available is True
        device = torch.device("cuda:0")
    else:
        sys.exit("Training must be done using GPUs.")

    toybox.set_seed(config["random_seed"])

    # for init wandb --------------------------------------------------
    package_name = 'wandb'
    flag_wandb = False
    try:
        import wandb
        #globals()[package_name] = __import__(package_name)
        print(f"'{package_name}' has been successfully imported.")
        flag_wandb = True
    except ImportError:
        print(f"Warning: '{package_name}' could not be imported. The program will continue.")       

    if flag_wandb == True:
        wandb.init(project="aoi",
                group=f"{model_name}",
                name=f"{config['runtime_name']}",
                config=config)

    # for loading data =======================================================
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
        num_workers=config["num_workers"],
    )

    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=collateFn
    )


    # for model =======================================================
    print("Initializing model...")
    if model_name == "gradtts":
        model = GradTTS.build_model(config, train_dataset.get_vocab_size())
    elif model_name == "gradseptts":
        model = GradSepTTS.build_model(config, train_dataset.get_vocab_size())
    elif model_name == "gradtfktts":
        model = GradTFKTTS.build_model(config, train_dataset.get_vocab_size())
    elif model_name == "gradtfk5tts":
        model = GradTFK5TTS.build_model(config, train_dataset.get_vocab_size())
    elif model_name == "gradtfkfultts":
        model = GradTFKFULTTS.build_model(config, train_dataset.get_vocab_size())
    elif model_name == "gradtimektts":
        model = GradTimeKTTS.build_model(config, train_dataset.get_vocab_size())
    else:
        raise ValueError(f"Error: '{model_name}' is not supported")
    print('Number of encoder + duration predictor parameters: %.2fm' % (model.encoder.nparams/1e6))
    print('Number of decoder parameters: %.2fm' % (model.decoder.nparams/1e6))
    print('Total parameters: %.2fm' % (model.nparams/1e6))
    if flag_wandb == True:
        wandb.watch(model, log_freq=100)

    start_epoch = 1
    start_steps = 1

    """
    if config["ckpt"]:
        print("loading ", config["ckpt"])
        epoch, steps, state_dict = torch.load(config["ckpt"], map_location="cpu")
        start_epoch = epoch + 1
        start_steps = steps + 1
        
    """
    latest_ckpt_path=toybox.find_latest_ckpt(config["ckpt"], model_name)
    if latest_ckpt_path:
        print(f"Latest checkpoint file: {latest_ckpt_path}")
        epoch, steps, state_dict = torch.load(latest_ckpt_path, map_location="cpu")
        start_epoch = epoch + 1
        start_steps = steps + 1
        model.load_state_dict(state_dict)
    else:
        #sys.exit("No checkpoint files found.")
        print("Warning: No checkpoint files found.")

    model = model.cuda()

    # for optimizer =======================================================
    print("Initializing optimizer...")
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config["learning_rate"])

    # for logger ==========================================================
    print("Initializing logger...")
    logger = SummaryWriter(log_dir=log_dir)

    ckpt_dir = log_dir / "ckpt"
    pic_dir = log_dir / "pic"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    pic_dir.mkdir(parents=True, exist_ok=True)

    # for train =======================================================
    print("Start training...")
    iteration = start_steps
    out_size = config["out_size"] * config["sample_rate"] // config["hop_size"]
    best_dur_loss = float('inf')
    best_prior_loss = float('inf')
    best_diffusion_loss = float('inf')

    for epoch in range(start_epoch, start_epoch + config["epoch_interval"]):
        model.train()
        dur_losses = []
        prior_losses = []
        diff_losses = []
        with tqdm(
            train_loader, total=len(train_dataset) // config["batch_size"]
        ) as progress_bar:
            for batch_idx, batch in enumerate(progress_bar):
                model.zero_grad()
                x, x_lengths = batch["x"].cuda(), batch["x_lengths"].cuda()
                y, y_lengths = batch["y"].cuda(), batch["y_lengths"].cuda()
                dur_loss, prior_loss, diff_loss = model.compute_loss(
                    x, x_lengths, y, y_lengths, out_size=out_size
                )
                loss = sum([dur_loss, prior_loss, diff_loss])
                loss.backward()

                enc_grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.encoder.parameters(), max_norm=1
                )
                dec_grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.decoder.parameters(), max_norm=1
                )
                optimizer.step()

                logger.add_scalar(
                    "training/duration_loss", dur_loss.item(), global_step=iteration
                )
                logger.add_scalar(
                    "training/prior_loss", prior_loss.item(), global_step=iteration
                )
                logger.add_scalar(
                    "training/diffusion_loss", diff_loss.item(), global_step=iteration
                )
                logger.add_scalar(
                    "training/encoder_grad_norm", enc_grad_norm, global_step=iteration
                )
                logger.add_scalar(
                    "training/decoder_grad_norm", dec_grad_norm, global_step=iteration
                )

                dur_losses.append(dur_loss.item())
                prior_losses.append(prior_loss.item())
                diff_losses.append(diff_loss.item())

                if batch_idx % 5 == 0:
                    msg = (
                        f"f{model_name} Epoch: {epoch}, iteration: {iteration} | "
                        f" dur_loss: {dur_loss.item()}, "
                        f"prior_loss: {prior_loss.item()}, "
                        f"diff_loss: {diff_loss.item()}"
                    )
                    progress_bar.set_description(msg)

                iteration += 1
                # for alert by wandb. --------------------------------------------------------
                if (flag_wandb == True) and (iteration % 100000 == 0):
                    wandb.alert(
                        title=f'fin train aoi {model_name} each epoch',
                        text=f'<@U052F2QKFMK> learning Now (> w <)q seq epoch:{epoch}, iteration:{iteration}',
                        level=wandb.AlertLevel.INFO
                    )

                # for save .pt ==================================================================
                if np.log10(iteration).is_integer() == True:
                    torch.save(
                        [epoch, iteration, model.state_dict()],
                        f=ckpt_dir / f"{model_name}_{epoch}_{iteration}.pt",
                    )
                
                if iteration >= config["max_step"]:
                    torch.save(
                        [epoch, iteration, model.state_dict()],
                        f=ckpt_dir / f"{model_name}_{epoch}_{iteration}.pt",
                    )
                    

                # for restore by wandb. --------------------------------------------------------
                if flag_wandb == True:
                    wandb.log(
                        {"epoch": epoch,
                        "dur_loss": dur_loss.item(),
                        "prior_loss": prior_loss.item(),
                        "diff_loss": diff_loss.item(),
                        }
                    )

        # for eval =======================================================
        model.eval()
        with torch.no_grad():
            all_dur_loss = []
            all_prior_loss = []
            all_diffusion_loss = []
            for _, item in enumerate(val_loader):
                x, x_lengths = batch["x"].cuda(), batch["x_lengths"].cuda()
                y, y_lengths = batch["y"].cuda(), batch["y_lengths"].cuda()

                dur_loss, prior_loss, diff_loss = model.compute_loss(
                    x, x_lengths, y, y_lengths, out_size=out_size
                )
                loss = sum([dur_loss, prior_loss, diff_loss])
                all_dur_loss.append(dur_loss)
                all_prior_loss.append(prior_loss)
                all_diffusion_loss.append(diff_loss)
                
            average_dur_loss = sum(all_dur_loss) / len(all_dur_loss)
            average_prior_loss = sum(all_prior_loss) / len(all_prior_loss)
            average_diffusion_loss = sum(all_diffusion_loss) / len(all_diffusion_loss)

            logger.add_scalar("val/duration_loss", average_dur_loss, global_step=epoch)
            logger.add_scalar("val/prior_loss", average_prior_loss, global_step=epoch)
            logger.add_scalar(
                "val/diffusion_loss", average_diffusion_loss, global_step=epoch
            )
            
            print(
                f"val duration_loss: {average_dur_loss}, "
                f"prior_loss: {average_prior_loss}, "
                f"diffusion_loss: {average_diffusion_loss}"
            )

            # for store best score
            if (average_diffusion_loss < best_diffusion_loss or
                average_dur_loss < best_dur_loss or
                average_prior_loss < best_prior_loss):

                if average_diffusion_loss < best_diffusion_loss:
                    best_diffusion_loss = average_diffusion_loss
                
                if average_dur_loss < best_dur_loss:
                    best_dur_loss = average_dur_loss
                
                if average_prior_loss < best_prior_loss:
                    best_prior_loss = average_prior_loss

                torch.save(
                    [epoch, iteration, model.state_dict()],
                    f=ckpt_dir / f"{model_name}_{epoch}_{iteration}.pt"
                )

                print(
                    f"New best model saved: "
                    f"duration_loss: {best_dur_loss}, "
                    f"prior_loss: {best_prior_loss}, "
                    f"diffusion_loss: {best_diffusion_loss}"
                )

            # for restore by wandb. --------------------------------------------------------
            if flag_wandb == True:
                wandb.log(
                    {"val_dur_loss": average_dur_loss.item(),
                    "val_prior_loss": average_prior_loss.item(),
                    "val_diff_loss": average_diffusion_loss.item(),
                    }
                )
            # for infer during validation step =======================================================
            y_enc, y_dec, attn = model(x, x_lengths, n_timesteps=config["n_timestep4infer"])
            idx = random.randrange(0, y_enc.shape[0])
            y_enc = y_enc[idx].cpu()
            y_dec = y_dec[idx].cpu()
            y = y[idx].cpu()
            attn = attn[idx][0].cpu()
            logger.add_image(
                "image/generated_enc",
                plot_tensor(y_enc),
                global_step=epoch,
                dataformats="HWC",
            )
            logger.add_image(
                "image/generated_dec",
                plot_tensor(y_dec),
                global_step=epoch,
                dataformats="HWC",
            )
            logger.add_image(
                "image/alignment",
                plot_tensor(attn),
                global_step=epoch,
                dataformats="HWC",
            )
            logger.add_image(
                "image/ground_truth",
                plot_tensor(y),
                global_step=epoch,
                dataformats="HWC",
            )

            save_plot(y_enc, pic_dir / f"generated_enc_{epoch}.png")
            save_plot(y_dec, pic_dir / f"generated_dec_{epoch}.png")
            save_plot(attn, pic_dir / f"alignment_{epoch}.png")
            save_plot(y, pic_dir / f"ground_truth_{epoch}.png")
        torch.save(
            [epoch, iteration, model.state_dict()],
            f=ckpt_dir / f"{model_name}_{epoch}_{iteration}.pt",
        )

        # for alert by wandb. --------------------------------------------------------
        """
        if flag_wandb == True:
            wandb.alert(
                title='fin valid aoi each epoch',
                text=f'<@U052F2QKFMK> fin valid seq (> w <)',
                level=wandb.AlertLevel.INFO
            )
        """


if flag_wandb == True:
    # for alert by wandb. --------------------------------------------------------
    wandb.alert(
        title='fin TRAIN aoi each epoch',
        text=f'<@U052F2QKFMK> fin valid seq (> w <)',
        level=wandb.AlertLevel.INFO
    )
    wandb.finish()

print('fin')
