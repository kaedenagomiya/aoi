import torch
import torch.nn.functional as F
#from torch.autograd import Variable
import numpy as np
from scipy.signal import get_window
from librosa.util import pad_center, tiny
from librosa.filters import mel as librosa_mel_fn
from librosa import stft, istft

from audio.audio_processing import (
    dynamic_range_compression,
    dynamic_range_decompression,
    window_sumsquare,
)


class STFT(torch.nn.Module):
    """
    This module implements an STFT using 1D convolution and 1D transpose convolutions.
    This is a bit tricky so there are some cases that probably won't work as working
    out the same sizes before and after in all overlap add setups is tough. Right now,
    this code should work with hop lengths that are half the filter length (50% overlap
    between frames).
    
    Keyword Arguments:
        filter_length {int}: Length of filters used (default: {1024})
        hop_length {int}: Hop length of STFT (restrict to 50% overlap between frames) (default: {512})
        win_length {[type]}: Length of the window function applied to each frame (if not specified, it
            equals the filter length). (default: {None})
        window {str}: Type of window to use (options are bartlett, hann, hamming, blackman, blackmanharris) 
            (default: {'hann'})

    cf.based from:
    - Prem Seetharaman's https://github.com/pseeth/pytorch-stft.
    - Implement Grad-TTS (switching process for device)
    """
    
    def __init__(self, 
                 filter_length=1024, 
                 hop_length=512, 
                 win_length=None,
                 window="hann"):
        
        super(STFT, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.forward_transform = None
        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack(
            [np.real(fourier_basis[:cutoff, :]),
             np.imag(fourier_basis[:cutoff, :])]
        )

        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        inverse_basis = torch.FloatTensor(
            np.linalg.pinv(scale * fourier_basis).T[:, None, :]
        )

        if window is not None:
            assert(filter_length >= win_length)
            # get window and zero center pad it to filter_length
            fft_window = get_window(window, win_length, fftbins=True)
            fft_window = pad_center(fft_window, filter_length)
            fft_window = torch.from_numpy(fft_window).float()

            # window the bases
            forward_basis *= fft_window
            inverse_basis *= fft_window

        self.register_buffer("forward_basis", forward_basis.float())
        self.register_buffer("inverse_basis", inverse_basis.float())

    def transform(self, input_data):
        """
        Take input data (audio) to STFT domain.
        
        Arguments:
            input_data
        
        Returns:
            magnitude: Magnitude of STFT with shape (num_batch, 
                num_frequencies, num_frames)
            phase: Phase of STFT with shape (num_batch, 
                num_frequencies, num_frames)
        """

        num_batches = input_data.size(0)
        num_samples = input_data.size(1)

        self.num_samples = num_samples

        if input_data.device.type == "cuda":
            # similar to librosa, reflect-pad the input
            input_data = input_data.view(num_batches, 1, num_samples)
            input_data = F.pad(
                input_data.unsqueeze(1),
                (int(self.filter_length / 2), int(self.filter_length / 2), 0, 0),
                mode='reflect')
            input_data = input_data.squeeze(1)
            
            forward_transform = F.conv1d(
                input_data,
                self.forward_basis,
                stride=self.hop_length,
                padding=0)

            cutoff = int((self.filter_length / 2) + 1)
            real_part = forward_transform[:, :cutoff, :]
            imag_part = forward_transform[:, cutoff:, :]
        else:
            x = input_data.detach().numpy()
            real_part = []
            imag_part = []
            for y in x:
                y_ = stft(y, self.filter_length, self.hop_length, self.win_length, self.window)
                real_part.append(y_.real[None,:,:])
                imag_part.append(y_.imag[None,:,:])
            real_part = np.concatenate(real_part, 0)
            imag_part = np.concatenate(imag_part, 0)
            
            real_part = torch.from_numpy(real_part).to(input_data.dtype)
            imag_part = torch.from_numpy(imag_part).to(input_data.dtype)

        magnitude = torch.sqrt(real_part**2 + imag_part**2)
        phase = torch.atan2(imag_part.data, real_part.data)

        return magnitude, phase
    
    def  inverse(self, magnitude, phase):
        """
        Call the inverse STFT (iSTFT), given magnitude and phase tensors produced 
        by the ```transform``` function.
        
        Arguments:
            magnitude: Magnitude of STFT with shape (num_batch, 
                num_frequencies, num_frames)
            phase: Phase of STFT with shape (num_batch, 
                num_frequencies, num_frames)
        Returns:
            inverse_transform: Reconstructed audio given magnitude and phase. Of
                shape (num_batch, num_samples)
        """
        recombine_magnitude_phase = torch.cat(
            [magnitude*torch.cos(phase), 
             magnitude*torch.sin(phase)],
             dim=1)
        
        if magnitude.device.type == "cuda":
            inverse_transform = F.conv_transpose1d(
                recombine_magnitude_phase,
                self.inverse_basis,
                stride=self.hop_length,
                padding=0
            )

            if self.window is not None:
                window_sum = window_sumsquare(
                    self.window,
                    magnitude.size(-1),
                    hop_length=self.hop_length,
                    win_length=self.win_length,
                    n_fft=self.filter_length,
                    dtype=np.float32
                )
                # remove modulation effects
                approx_nonzero_indices = torch.from_numpy(
                    np.where(window_sum > tiny(window_sum))[0]
                )
                window_sum = torch.from_numpy(window_sum).to(inverse_transform.device)
                inverse_transform[:, :, approx_nonzero_indices] /= window_sum[approx_nonzero_indices]

                # scale by hop ratio
                inverse_transform *= float(self.filter_length) / self.hop_length

            inverse_transform = inverse_transform[:, :, int(self.filter_length/2):]
            inverse_transform = inverse_transform[:, :, :-int(self.filter_length/2):]
            inverse_transform = inverse_transform.squeeze(1)
        else:
            x_org = recombine_magnitude_phase.detach().numpy()
            n_b, n_f, n_t = x_org.shape
            x = np.empty([n_b, n_f//2, n_t], dtype=np.complex64)
            x.real = x_org[:,:n_f//2]
            x.imag = x_org[:,n_f//2:]
            inverse_transform = []
            for y in x:
                y_ = istft(y, self.hop_length, self.win_length, self.window)
                inverse_transform.append(y_[None,:])
            inverse_transform = np.concatenate(inverse_transform, 0)
            inverse_transform = torch.from_numpy(inverse_transform).to(recombine_magnitude_phase.dtype)

        return inverse_transform

    def forward(self, input_data):
        """
        Take input data (audio) to STFT domain and then back to audio.
        
        Arguments:
            input_data: Tensor of floats, with shape (num_batch, num_samples)
        
        Returns:
            reconstruction: Reconstructed audio given magnitude and phase. Of
                shape (num_batch, num_samples)
        """
        self.magnitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction


class TacotronSTFT(torch.nn.Module):
    """
    cf.based from:
    - Prem Seetharaman's https://github.com/pseeth/pytorch-stft.
    - Implement Grad-TTS(in commons.py), DDGAN.
    """
    def __init__(
            self,
            filter_length=1024,
            hop_length=256,
            win_length=1024,
            n_mel_channels=80,
            sampling_rate=22050,
            mel_fmin=0.0,
            mel_fmax=8000.0
        ):
        super(TacotronSTFT, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.stft_fn = STFT(filter_length, hop_length, win_length)
        mel_basis = librosa_mel_fn(
            sampling_rate, filter_length, n_mel_channels, mel_fmin, mel_fmax
        )
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)

    def spectral_normalize(self, magnitudes):
        output = dynamic_range_compression(magnitudes)
        return output
    
    def spectral_de_normalize(self, magnitudes):
        output = dynamic_range_decompression(magnitudes)
        return output
    
    def mel_spectrogram(self, y):
        """Computes mel-spectrograms from a batch of waves
        
        Arguments:
            y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]

        Returns:
            mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
        """
        assert(torch.min(y.data) >= -1)
        assert(torch.max(y.data) <= 1)

        magnitudes, phases = self.stft_fn.transform(y)
        magnitudes = magnitudes.data
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = self.spectral_normalize(mel_output)
        return mel_output
