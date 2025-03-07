# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import math
import random

import torch
import monotonic_align

from .base import BaseModule
from .text_encoder import TextEncoder
from .diffusion import Diffusion
from .utils import (sequence_mask,
                    generate_path,
                    duration_loss,
                    fix_len_compatibility)

class GradTFKFastTTS(BaseModule):

    def __init__(self, n_vocab, n_spks, spk_emb_dim, n_enc_channels,
                 filter_channels, filter_channels_dp, n_heads, n_enc_layers,
                 enc_kernel, enc_dropout, window_size, n_feats, dec_dim,
                 beta_min, beta_max, pe_scale):
        super().__init__()
        self.n_vocab = n_vocab
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.n_enc_channels = n_enc_channels
        self.filter_channels = filter_channels
        self.filter_channels_dp = filter_channels_dp
        self.n_heads = n_heads
        self.n_enc_layers = n_enc_layers
        self.enc_kernel = enc_kernel
        self.enc_dropout = enc_dropout
        self.window_size = window_size
        self.n_feats = n_feats
        self.dec_dim = dec_dim
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.pe_scale = pe_scale

        if n_spks > 1:
            self.spk_emb = torch.nn.Embedding(n_spks, spk_emb_dim)
        self.encoder = TextEncoder(n_vocab, n_feats, n_enc_channels,
                                   filter_channels, filter_channels_dp,
                                   n_heads, n_enc_layers, enc_kernel,
                                   enc_dropout, window_size)
        self.decoder = Diffusion(n_feats, dec_dim, n_spks, spk_emb_dim,
                                 beta_min, beta_max, pe_scale)

    @classmethod
    def build_model(cls, config, vocab_size):
        return cls(vocab_size, 1, None, config['n_enc_channels'],
                   config['filter_channels'], config['filter_channels_dp'],
                   config['n_heads'], config['n_enc_layers'],
                   config['enc_kernel'], config['enc_dropout'],
                   config['window_size'], config['n_mels'], config['dec_dim'],
                   config['beta_min'], config['beta_max'], config['pe_scale'])

    @torch.no_grad()
    def forward(self,
                x,
                x_lengths,
                n_timesteps,
                temperature=1.0,
                stoc=False,
                spk=None,
                length_scale=1.0,
                solver='original'):
        """
        Generates mel-spectrogram from text. Returns:
            1. encoder outputs
            2. decoder outputs
            3. generated alignment
        
        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
            x_lengths (torch.Tensor): lengths of texts in batch.
            n_timesteps (int): number of steps to use for reverse diffusion in decoder.
            temperature (float, optional): controls variance of terminal distribution.
            stoc (bool, optional): flag that adds stochastic term to the decoder sampler.
                Usually, does not provide synthesis improvements.
            length_scale (float, optional): controls speech pace.
                Increase value to slow down generated speech and vice versa.
        """
        
        x, x_lengths = self.relocate_input([x, x_lengths])

        if self.n_spks > 1:
            # Get speaker embedding
            spk = self.spk_emb(spk)

        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask = self.encoder(x, x_lengths, spk)

        #w = torch.exp(logw) * x_mask
        w = (torch.exp(logw) - 1) * x_mask
        w_ceil = torch.ceil(w * length_scale)
        # y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_lengths = torch.clamp(torch.sum(w_ceil, [1, 2]), min=0).long()
        y_max_length = int(y_lengths.max())
        y_max_length_ = fix_len_compatibility(y_max_length)

        # Using obtained durations `w` construct alignment map `attn`
        y_mask = sequence_mask(y_lengths,
                               y_max_length_).unsqueeze(1).to(x_mask.dtype)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        attn = generate_path(w_ceil.squeeze(1),
                             attn_mask.squeeze(1)).unsqueeze(1)

        # Align encoded text and get mu_y
        mu_y = torch.matmul(
            attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)
        encoder_outputs = mu_y[:, :, :y_max_length]

        # Sample latent representation from terminal distribution N(mu_y, I)
        z = mu_y + torch.randn_like(mu_y, device=mu_y.device) / temperature
        # Generate sample by performing reverse dynamics
        decoder_outputs = self.decoder(z, y_mask, mu_y, n_timesteps, stoc, spk,
                                       solver)
        decoder_outputs = decoder_outputs[:, :, :y_max_length]

        return encoder_outputs, decoder_outputs, attn[:, :, :y_max_length]


    @torch.no_grad()
    def forward_streaming(self,
                          x,
                          x_lengths,
                          n_timesteps,
                          temperature=1.0,
                          stoc=False,
                          spk=None,
                          length_scale=1.0,
                          out_size=None,
                          solver='original'):
        #    if chunk_method == 'simple':
        #        return self.forward_streaming_simple_chunk(x, x_lengths,
        #                                                   n_timesteps,
        #                                                   temperature, stoc, spk,
        #                                                   length_scale, out_size,
        #                                                   solver)
        #    elif chunk_method == 'padding':
        #        return self.forward_streaming_padding_chunk(
        #            x, x_lengths, n_timesteps, temperature, stoc, spk,
        #            length_scale, out_size, solver)
        #    else:
        #        raise ValueError(f'Wrong chunk method: {chunk_method}!')

        #@torch.no_grad()
        #def forward_streaming_padding_chunk(self, x, x_lengths, n_timesteps,
        #                                    temperature, stoc, spk, length_scale,
        #                                    out_size, solver):
        """
        Generates mel-spectrogram from text. Returns:
            1. encoder outputs
            2. decoder outputs
            3. generated alignment
        
        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
            x_lengths (torch.Tensor): lengths of texts in batch.
            n_timesteps (int): number of steps to use for reverse diffusion in decoder.
            temperature (float, optional): controls variance of terminal distribution.
            stoc (bool, optional): flag that adds stochastic term to the decoder sampler.
                Usually, does not provide synthesis improvements.
            length_scale (float, optional): controls speech pace.
                Increase value to slow down generated speech and vice versa.
        """
        x, x_lengths = self.relocate_input([x, x_lengths])
        assert x.shape[0] == 1  # streaming inference only support batch size 1
        if self.n_spks > 1:
            # Get speaker embedding
            spk = self.spk_emb(spk)

        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask = self.encoder(x, x_lengths, spk)
        w = (torch.exp(logw) - 1) * x_mask
        w_ceil = torch.ceil(w * length_scale).squeeze(1)
        y_lengths = torch.clamp(torch.sum(w_ceil, dim=1), min=0).long()
        y_max_length = int(y_lengths.max())
        y_max_length_ = fix_len_compatibility(y_max_length)
        out_size = fix_len_compatibility(out_size)

        # Using obtained durations `w` construct alignment map `attn`
        y_mask = sequence_mask(y_lengths,
                               y_max_length_).unsqueeze(1).to(x_mask.dtype)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        attn = generate_path(w_ceil, attn_mask.squeeze(1))
        (num_chunks, chunk_lengths, start_frames, end_frames, lpad,
         rpad) = split_durs(w_ceil, out_size)
        for i in range(num_chunks):
            lp = lpad[i]
            rp = rpad[i]
            l = chunk_lengths[i]

            # start_idx should be divisible by downsampling factor
            start_idx = fix_len_compatibility(start_frames[i] - lp,
                                              type='floor')
            # adjust left padding part according to start_idx
            lp += start_frames[i] - lp - start_idx
            end_idx = min(y_max_length_,
                          fix_len_compatibility(end_frames[i] + rp))

            y_mask_cut = y_mask[:, :, start_idx:end_idx]
            attn_cut = attn[:, :, start_idx:end_idx]
            mu_y_cut = torch.matmul(attn_cut.transpose(1, 2),
                                    mu_x.transpose(1, 2))
            mu_y_cut = mu_y_cut.transpose(1, 2)
            z_cut = mu_y_cut + torch.randn_like(
                mu_y_cut, device=mu_y_cut.device) / temperature
            decoder_output_cut = self.decoder(z_cut, y_mask_cut, mu_y_cut,
                                              n_timesteps, stoc, spk, solver)
            yield (mu_y_cut[:, :, lp:lp + l],
                   decoder_output_cut[:, :, lp:lp + l], attn_cut[:, :,
                                                                 lp:lp + l])


    def compute_loss(self,
                     x,
                     x_lengths,
                     y,
                     y_lengths,
                     spk=None,
                     out_size=None):
        """
        Computes 3 losses:
            1. duration loss: loss between predicted token durations and those extracted by Monotinic Alignment Search (MAS).
            2. prior loss: loss between mel-spectrogram and encoder outputs.
            3. diffusion loss: loss between gaussian noise and its reconstruction by diffusion-based decoder.
            
        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
            x_lengths (torch.Tensor): lengths of texts in batch.
            y (torch.Tensor): batch of corresponding mel-spectrograms.
            y_lengths (torch.Tensor): lengths of mel-spectrograms in batch.
            out_size (int, optional): length (in mel's sampling rate) of segment to cut, on which decoder will be trained.
                Should be divisible by 2^{num of UNet downsamplings}. Needed to increase batch size.
        """
        out_size = fix_len_compatibility(out_size)
        x, x_lengths, y, y_lengths = self.relocate_input(
            [x, x_lengths, y, y_lengths])

        if self.n_spks > 1:
            # Get speaker embedding
            spk = self.spk_emb(spk)

        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask = self.encoder(x, x_lengths, spk)
        # mu_x: (b,80,tx)
        # logw: (b,1,tx)
        # y: (b,80,ty)
        y_max_length = y.shape[-1]

        y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)

        # Use MAS to find most likely alignment `attn` between text and mel-spectrogram
        with torch.no_grad():
            # sum(-0.5*log(2*pi*sigma_i^2))
            const = -0.5 * math.log(2 * math.pi) * self.n_feats
            # factor: (b,80,tx)
            factor = -0.5 * torch.ones(
                mu_x.shape, dtype=mu_x.dtype, device=mu_x.device)
            # y_square: (b,tx,ty), y_square_{i,j}: mu_i is aligned with y_j
            # sum(-0.5*y_i^2*sigma^(-2))
            y_square = torch.matmul(factor.transpose(1, 2), y**2)
            # y_mu_double: (b,tx,ty), sum(y_i*mu_i*sigma_i^(-2))
            y_mu_double = torch.matmul(mu_x.transpose(1, 2), y)
            # mu_square: (b,tx,1), -0.5*sum(mu_i^2*sigma_i^(-2))
            mu_square = torch.sum(factor * (mu_x**2), 1).unsqueeze(-1)
            log_prior = y_square - y_mu_double + mu_square + const

            #attn = monotonic_align.maximum_path(log_prior, attn_mask.squeeze(1))
            attn = monotonic_align.maximum_path(
                log_prior.permute(0, 2, 1),
                attn_mask.squeeze(1).permute(0, 2, 1)).permute(0, 2, 1)
            # attn: (b,tx,ty)
            # attn = attn.detach()

        # Compute loss between predicted log-scaled durations and those obtained from MAS
        # logw_ = torch.log(1e-8 + torch.sum(attn.unsqueeze(1), -1)) * x_mask
        logw_ = torch.log(1 + torch.sum(attn.unsqueeze(1), -1)) * x_mask
        # logw_ = torch.log(1 + torch.sum(attn.unsqueeze(1), -1)) * x_mask
        dur_loss = duration_loss(logw, logw_, x_lengths)

        # Cut a small segment of mel-spectrogram in order to increase batch size
        if not isinstance(out_size, type(None)):
            max_offset = (y_lengths - out_size).clamp(0)
            offset_ranges = list(
                zip([0] * max_offset.shape[0],
                    max_offset.cpu().numpy()))
            out_offset = torch.LongTensor([
                torch.tensor(
                    random.choice(range(start, end)) if end > start else 0)
                for start, end in offset_ranges
            ]).to(y_lengths)

            attn_cut = torch.zeros(attn.shape[0],
                                   attn.shape[1],
                                   out_size,
                                   dtype=attn.dtype,
                                   device=attn.device)
            y_cut = torch.zeros(y.shape[0],
                                self.n_feats,
                                out_size,
                                dtype=y.dtype,
                                device=y.device)
            y_cut_lengths = []
            for i, (y_, out_offset_) in enumerate(zip(y, out_offset)):
                y_cut_length = out_size + (y_lengths[i] - out_size).clamp(
                    None, 0)
                y_cut_lengths.append(y_cut_length)
                cut_lower, cut_upper = out_offset_, out_offset_ + y_cut_length
                y_cut[i, :, :y_cut_length] = y_[:, cut_lower:cut_upper]
                attn_cut[i, :, :y_cut_length] = attn[i, :, cut_lower:cut_upper]
            y_cut_lengths = torch.LongTensor(y_cut_lengths)
            y_cut_mask = sequence_mask(y_cut_lengths).unsqueeze(1).to(y_mask)

            attn = attn_cut
            y = y_cut
            y_mask = y_cut_mask

        # Align encoded text with mel-spectrogram and get mu_y segment
        mu_y = torch.matmul(attn.transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)
        # mu_y: (b,80,t_y_clip)
        # Compute loss of score-based decoder
        diff_loss, xt = self.decoder.compute_loss(y, y_mask, mu_y, spk)

                # Compute loss between aligned encoder outputs and mel-spectrogram
        # prior_loss: sum(0.5*log(2*pi*sigma_i^2)+0.5*(y_i-mu_i)^2*sigma_i^(-2))
        prior_loss = torch.sum(0.5 * ((y - mu_y)**2 + math.log(2 * math.pi)) *
                               y_mask)
        prior_loss = prior_loss / (torch.sum(y_mask) * self.n_feats)

        return dur_loss, prior_loss, diff_loss


def split_durs(durations, chunk_frames):
    """

    Args:
        durations (_type_): duration for each token. Shape (1,tx)
        chunk_frames (_type_): frames per chunk.

    """
    durations = durations.flatten()
    cum_sum = durations.cumsum(dim=0).int()

    idx = torch.div(cum_sum, chunk_frames, rounding_mode='trunc').int()
    start_token_idx = 0
    num_chunks = idx.max() + 1
    lengths = torch.zeros((num_chunks),
                          device=durations.device,
                          dtype=torch.int)
    lpad = torch.zeros_like(lengths, device=lengths.device)
    rpad = torch.zeros_like(lengths, device=lengths.device)
    for i in range(num_chunks):
        duration_chunk_mask = i == idx
        duration_chunk_tokens = duration_chunk_mask.sum()
        duration_chunk = durations * duration_chunk_mask
        duration_chunk_frames = duration_chunk.sum()
        if i > 0:
            lpad[i] = durations[start_token_idx - 1]
        if i < num_chunks - 1:
            rpad[i] = durations[start_token_idx + duration_chunk_tokens]
        lengths[i] = duration_chunk_frames
        start_token_idx += duration_chunk_tokens
    end_frames = lengths.cumsum(dim=0)
    start_frames = end_frames - lengths
    return num_chunks, lengths, start_frames, end_frames, lpad, rpad