# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import math
import torch
from torch import nn

from .layer import (ResnetBlock,
                    LinearAttention)
from .base import BaseModule

class Mish(BaseModule):

    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))


class Upsample(BaseModule):

    def __init__(self, dim):
        super(Upsample, self).__init__()
        self.conv = torch.nn.ConvTranspose2d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Downsample(BaseModule):

    def __init__(self, dim):
        super(Downsample, self).__init__()
        self.conv = torch.nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class ConvBlock(BaseModule):

    def __init__(self, dim, dim_out, groups=8):
        super(ConvBlock, self).__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(dim, dim_out, 3, padding=1),
            torch.nn.GroupNorm(groups, dim_out), Mish())

    def forward(self, x, mask):
        output = self.block(x * mask)
        return output * mask


class Rezero(BaseModule):

    def __init__(self, fn):
        super(Rezero, self).__init__()
        self.fn = fn
        self.g = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.fn(x) * self.g


class Residual(BaseModule):

    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        output = self.fn(x, *args, **kwargs) + x
        return output
    

class SinusoidalPosEmb(BaseModule):

    def __init__(self, dim):
        super(SinusoidalPosEmb, self).__init__()
        self.dim = dim

    def forward(self, x, scale=1000):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class GradLogPEstimator2d(BaseModule):

    def __init__(self,
                 dim,
                 dim_mults=(1, 2, 4),
                 groups=8,
                 n_spks=None,
                 spk_emb_dim=64,
                 n_feats=80,
                 pe_scale=1000):
        super(GradLogPEstimator2d, self).__init__()
        self.dim = dim  # 64
        self.dim_mults = dim_mults  # (1,2,4)
        self.groups = groups  # 8
        self.n_spks = n_spks if not isinstance(n_spks, type(None)) else 1  # 1
        self.spk_emb_dim = spk_emb_dim  # None
        self.pe_scale = pe_scale  # 1000

        if n_spks > 1:
            self.spk_mlp = torch.nn.Sequential(
                torch.nn.Linear(spk_emb_dim, spk_emb_dim * 4), Mish(),
                torch.nn.Linear(spk_emb_dim * 4, n_feats))
        self.time_pos_emb = SinusoidalPosEmb(dim)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(dim, dim * 4), Mish(),
                                       torch.nn.Linear(dim * 4, dim), Mish())

        dims = [
            2 + (1 if n_spks > 1 else 0), *map(lambda m: dim * m, dim_mults)
        ]  # [2,64,128,256]
        in_out = list(zip(dims[:-1], dims[1:]))  # [(2,64),(64,128),(128,256)]
        self.downs = torch.nn.ModuleList([])
        self.ups = torch.nn.ModuleList([])
        num_resolutions = len(in_out)  # 3

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(
                torch.nn.ModuleList([
                    ResnetBlock(dim_in, dim_out, time_emb_dim=dim),
                    ResnetBlock(dim_out, dim_out, time_emb_dim=dim),
                    Residual(Rezero(LinearAttention(dim_out))),
                    Downsample(dim_out)
                    if not is_last else torch.nn.Identity()
                ]))

        mid_dim = dims[-1]  # 256
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim)
        self.mid_attn = Residual(Rezero(LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            self.ups.append(
                torch.nn.ModuleList([
                    ResnetBlock(dim_out * 2, dim_in, time_emb_dim=dim),
                    ResnetBlock(dim_in, dim_in, time_emb_dim=dim),
                    Residual(Rezero(LinearAttention(dim_in))),
                    Upsample(dim_in)
                ]))
        self.final_block = ConvBlock(dim, dim)
        self.final_conv = torch.nn.Conv2d(dim, 1, 1)

    def forward(self, x, mask, mu, t, spk=None):
        """
        Args:
            x (_type_): shape (b,80,tx)
            mask (_type_): shape (b,1,tx)
            mu (_type_): shape (b,80,tx)
            t (_type_): shape (b)
            spk (_type_, optional):
        """
        if not isinstance(spk, type(None)):
            s = self.spk_mlp(spk)

        t = self.time_pos_emb(t, scale=self.pe_scale)
        t = self.mlp(t)

        if self.n_spks < 2:
            x = torch.stack([mu, x], 1)
        else:
            s = s.unsqueeze(-1).repeat(1, 1, x.shape[-1])
            x = torch.stack([mu, x, s], 1)
        # mask: (b,1,tx)
        mask = mask.unsqueeze(1)
        hiddens = []
        masks = [mask]
        # x: (b,2,80,tx) -> (b,64,40,86) -> (b,128,20,43) -> (b,256,20,43)->mid
        # -> (b,256,20,43) -> (b,128,40,86) -> (b,64,80,172) -> (b,64,80,172) -> (b,1,80,172)
        for resnet1, resnet2, attn, downsample in self.downs:
            mask_down = masks[-1]
            x = resnet1(x, mask_down, t)
            x = resnet2(x, mask_down, t)
            x = attn(x)
            hiddens.append(x)
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, :, ::2])
        masks = masks[:-1]
        mask_mid = masks[-1]
        x = self.mid_block1(x, mask_mid, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, mask_mid, t)
        for resnet1, resnet2, attn, upsample in self.ups:
            mask_up = masks.pop()
            x = torch.cat((x, hiddens.pop()), dim=1)
            x = resnet1(x, mask_up, t)
            x = resnet2(x, mask_up, t)
            x = attn(x)
            x = upsample(x * mask_up)
        x = self.final_block(x, mask)
        output = self.final_conv(x * mask)

        return (output * mask).squeeze(1)


def get_noise(t, beta_init, beta_term, cumulative=False):
    if cumulative:
        # int(beta_0+(beta_1-beta_0)*t) = beta_0*t+0.5*(beta_1-beta_0)*t^2
        noise = beta_init * t + 0.5 * (beta_term - beta_init) * (t**2)
    else:
        noise = beta_init + (beta_term - beta_init) * t
    return noise


class Diffusion(BaseModule):

    def __init__(self,
                 n_feats,
                 dim,
                 n_spks=1,
                 spk_emb_dim=64,
                 beta_min=0.05,
                 beta_max=20,
                 pe_scale=1000):
        super(Diffusion, self).__init__()
        self.n_feats = n_feats
        self.dim = dim
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.pe_scale = pe_scale
        #self.dpm_solver_sch = NoiseScheduleVP()
        self.estimator = GradLogPEstimator2d(dim,
                                             n_spks=n_spks,
                                             spk_emb_dim=spk_emb_dim,
                                             pe_scale=pe_scale)
    
    def forward_diffusion(self, x0, mask, mu, t):
        """
        Args:
            x0 (_type_): shape (b,80,tx)
            mask (_type_): shape (b,1,tx)
            mu (_type_): shape (b,80,tx)
            t (_type_): shape (b)
        """
        time = t.unsqueeze(-1).unsqueeze(-1)  # (b,1,1)
        cum_noise = get_noise(time,
                              self.beta_min,
                              self.beta_max,
                              cumulative=True)
        mean = x0 * torch.exp(
            -0.5 * cum_noise) + mu * (1.0 - torch.exp(-0.5 * cum_noise))
        variance = 1.0 - torch.exp(-cum_noise)
        z = torch.randn(x0.shape,
                        dtype=x0.dtype,
                        device=x0.device,
                        requires_grad=False)
        xt = mean + z * torch.sqrt(variance)
        return xt * mask, z * mask

    @torch.no_grad()
    def reverse_diffusion_original(self,
                                   z,
                                   mask,
                                   mu,
                                   n_timesteps,
                                   stoc=False,
                                   spk=None):
        h = 1.0 / n_timesteps
        xt = z * mask
        for i in range(n_timesteps):
            t = (1.0 - (i + 0.5) * h) * torch.ones(
                z.shape[0], dtype=z.dtype, device=z.device)
            time = t.unsqueeze(-1).unsqueeze(-1)
            noise_t = get_noise(time,
                                self.beta_min,
                                self.beta_max,
                                cumulative=False)
            if stoc:  # adds stochastic term
                dxt_det = 0.5 * (mu - xt) - self.estimator(
                    xt, mask, mu, t, spk)
                dxt_det = dxt_det * noise_t * h
                dxt_stoc = torch.randn(z.shape,
                                       dtype=z.dtype,
                                       device=z.device,
                                       requires_grad=False)
                dxt_stoc = dxt_stoc * torch.sqrt(noise_t * h)
                dxt = dxt_det + dxt_stoc
            else:
                dxt = 0.5 * (mu - xt - self.estimator(xt, mask, mu, t, spk))
                dxt = dxt * noise_t * h
            xt = (xt - dxt) * mask
        return xt

    @torch.no_grad()
    def forward(self,
                z,
                mask,
                mu,
                n_timesteps,
                stoc=False,
                spk=None,
                solver='original'):
        if solver == 'original':
            return self.reverse_diffusion_original(z, mask, mu, n_timesteps,
                                                   stoc, spk)
        else:
            raise ValueError(f'Wrong solver:{solver}!')

    def loss_t(self, x0, mask, mu, t, spk=None):
        """
        Args:
            x0 (_type_): shape (b,80,tx)
            mask (_type_): shape (b,1,tx)
            mu (_type_): shape (b,80,tx)
            t (_type_): shape (b)
            spk (_type_, optional): 
        """
        xt, z = self.forward_diffusion(x0, mask, mu, t)
        time = t.unsqueeze(-1).unsqueeze(-1)
        cum_noise = get_noise(time,
                              self.beta_min,
                              self.beta_max,
                              cumulative=True)
        noise_estimation = self.estimator(xt, mask, mu, t, spk)
        noise_estimation *= torch.sqrt(1.0 - torch.exp(-cum_noise))
        loss = torch.sum(
            (noise_estimation + z)**2) / (torch.sum(mask) * self.n_feats)
        return loss, xt

    def compute_loss(self, x0, mask, mu, spk=None, offset=1e-5):
        """
        Args:
            x0 (_type_): shape (b,80,tx)
            mask (_type_): shape (b,1,tx)
            mu (_type_): shape (b,80,tx)
        """
        t = torch.rand(x0.shape[0],
                       dtype=x0.dtype,
                       device=x0.device,
                       requires_grad=False)
        t = torch.clamp(t, offset, 1.0 - offset)
        # t: (b)
        return self.loss_t(x0, mask, mu, t, spk)