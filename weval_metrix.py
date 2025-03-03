import numpy as np
import numpy.polynomial.polynomial as poly
import pysptk
import pyworld as pw

from scipy import spatial
from fastdtw import fastdtw
import soundfile as sf
import librosa
from pypesq import pesq

import onnxruntime as ort

# for MCD (Mel-cepstrogram Distance) ======================================================
def extract_mcd(
    x,
    fs,
    n_fft=512,
    n_hopsize=256,
    mcep_dim=25,
    mcep_alpha=0.41,
    is_padding=False,
):
    if is_padding:
        n_pad = n_fft - (len(x) - n_fft) % n_hopsize
        x = np.pad(x, (0, n_pad), "reflect")

    # get frames
    n_frame = (len(x) - n_fft) // n_hopsize + 1

    # get window func
    win = pysptk.sptk.hamming(n_fft)

    # check exist mcep_dim and mcep_alpha
    if mcep_dim is None or mcep_alpha is None:
        mcep_dim, mcep_alpha = _get_best_mcep_params(fs)

    # calc linear spectrogram
    mcep = [
        pysptk.mcep(
            x[n_hopsize * i : n_hopsize * i + n_fft] * win,
            mcep_dim,
            mcep_alpha,
            eps=1e-6,
            etype=1,
        )
        for i in range(n_frame)
    ]

    return np.stack(mcep)


def _get_best_mcep_params(fs):
    if fs == 16000:
        return 23, 0.42
    elif fs == 22050:
        return 34, 0.45
    else:
        raise ValueError(f"Not found the setting for {fs}.")
    
class MCD:
    # https://pysptk.readthedocs.io/en/latest/sptk.html#mel-generalized-cepstrum-analysis
    """
    if u use
    import toybox
    import weval_metrix
    samplerate = 16000
    metrics = weval_metrix.MCD(sr=samplerate)
    ref = toybox.load_wavtonp("data/ljspeech/LJSpeech-1.1/wavs/LJ017-0027.wav")
    synth = toybox.load_wavtonp("result4eval/infer4colb/gradtts/cpu/e500_n50/wav/LJ017-0027.wav")
    score = metrics.score(ref, synth)
    >>> score
    np.float64(5.601645565831899)
    """
    def __init__(self, sr=16000, n_fft=1024, n_hopsize=256, mcep_dim=None, mcep_alpha=None):
        self.sr = sr
        self.n_fft = n_fft
        self.n_hopsize = n_hopsize
        self.mcep_dim = mcep_dim
        self.mcep_alpha = mcep_alpha

    def score(self, gt_wav, gen_wav):
        gen_mcep = extract_mcd(
            x=gen_wav,
            fs=self.sr,
            n_fft=self.n_fft,
            n_hopsize=self.n_hopsize,
            mcep_dim=self.mcep_dim,
            mcep_alpha=self.mcep_alpha,
        )
        gt_mcep = extract_mcd(
            x=gt_wav,
            fs=self.sr,
            n_fft=self.n_fft,
            n_hopsize=self.n_hopsize,
            mcep_dim=self.mcep_dim,
            mcep_alpha=self.mcep_alpha,
        )

        # DTW
        _, path = fastdtw(gen_mcep, gt_mcep, dist=spatial.distance.euclidean)
        twf = np.array(path).T
        gen_mcep_dtw = gen_mcep[twf[0]]
        gt_mcep_dtw = gt_mcep[twf[1]]

        # MCD
        diff2sum = np.sum((gen_mcep_dtw - gt_mcep_dtw) ** 2, 1)
        mcd = np.mean(10.0 / np.log(10.0) * np.sqrt(2 * diff2sum), 0)
        return mcd


# for logf0rmse ======================================================
def extract_f0(
    x,
    fs,
    f0min=40,
    f0max = 800,
    n_fft = 512,
    n_hopsize = 256,
    mcep_dim = 25,
    mcep_alpha = 0.41,
):
    # by world
    # extract features
    x = x.astype(np.float64)
    f0, time_axis = pw.harvest(
        x,
        fs,
        f0_floor=f0min,
        f0_ceil=f0max,
        frame_period=n_hopsize / fs * 1000,
    )
    sp = pw.cheaptrick(x, f0, time_axis, fs, fft_size=n_fft)
    if mcep_dim is None or mcep_alpha is None:
        mcep_dim, mcep_alpha = _get_best_mcep_params(fs)
    mcep = pysptk.sp2mc(sp, mcep_dim, mcep_alpha)

    return mcep, f0


def _get_best_mcep_params(fs):
    if fs == 16000:
        return 23, 0.42
    elif fs == 22050:
        return 34, 0.45


class LogF0RMSE:
    """
    if u use
    import toybox
    import weval_metrix
    samplerate = 16000
    metrics = weval_metrix.LogF0RMSE(sr=samplerate)
    ref = toybox.load_wavtonp("data/ljspeech/LJSpeech-1.1/wavs/LJ017-0027.wav")
    synth = toybox.load_wavtonp("result4eval/infer4colb/gradtts/cpu/e500_n50/wav/LJ017-0027.wav")
    score = metrics.score(ref, synth)
    >>> score
    np.float64(0.2830700455129188)
    """

    def __init__(
        self,
        sr,
        f0min: int = 40,
        f0max: int = 800,
        n_fft: int = 512,
        n_hopsize: int = 256,
        mcep_dim: int = 25,
        mcep_alpha: float = 0.41):
        self.sr = sr
        self.f0min = f0min
        self.f0max = f0max
        self.n_fft = n_fft
        self.n_hopsize = n_hopsize
        self.mcep_dim = mcep_dim
        self.mcep_alpha = mcep_alpha
    
    def score(self, gt_wav, gen_wav):
        gen_mcep, gen_f0 = extract_f0(
            x=gen_wav,
            fs=self.sr,
            f0min=self.f0min,
            f0max=self.f0max,
            n_fft=self.n_fft,
            n_hopsize=self.n_hopsize,
            mcep_dim=self.mcep_dim,
            mcep_alpha=self.mcep_alpha,
        )
        gt_mcep, gt_f0 = extract_f0(
            x=gt_wav,
            fs=self.sr,
            f0min=self.f0min,
            f0max=self.f0max,
            n_fft=self.n_fft,
            n_hopsize=self.n_hopsize,
            mcep_dim=self.mcep_dim,
            mcep_alpha=self.mcep_alpha,
        )

        # DTW
        _, path = fastdtw(gen_mcep, gt_mcep, dist=spatial.distance.euclidean)
        twf = np.array(path).T
        gen_f0_dtw = gen_f0[twf[0]]
        gt_f0_dtw = gt_f0[twf[1]]

        # Get voiced part
        nonzero_idxs = np.where((gen_f0_dtw != 0) & (gt_f0_dtw != 0))[0]
        gen_f0_dtw_voiced = np.log(gen_f0_dtw[nonzero_idxs])
        gt_f0_dtw_voiced = np.log(gt_f0_dtw[nonzero_idxs])

        # log F0 RMSE
        log_f0_rmse = np.sqrt(np.mean((gen_f0_dtw_voiced - gt_f0_dtw_voiced) ** 2))
        return log_f0_rmse



# for pesq ======================================================
class PESQ:
    """
    if you use:
    import toybox
    import weval_metrix
    samplerate = 16000
    metrics = weval_metrix.PESQ(sr=samplerate)
    ref = toybox.load_wavtonp("data/ljspeech/LJSpeech-1.1/wavs/LJ017-0027.wav")
    synth = toybox.load_wavtonp("result4eval/infer4colb/gradtts/cpu/e500_n50/wav/LJ017-0027.wav")
    min_len = min(len(ref), len(synth))
    ref = ref[:min_len]
    synth = synth[:min_len]
    score = metrics.score(ref, synth)
    >>> score
    0.2968555986881256
    """
    def __init__(self, sr=16000):
        self.sr = sr
        self.tar_fs = 16000
    
    def score(self, gt_wav, gen_wav):
        if self.sr != self.tar_fs:
            gt_wav = librosa.resample(gt_wav.astype(np.float64), self.sr, self.tar_fs)
        if self.sr != self.tar_fs:
            gen_wav = librosa.resample(gen_wav.astype(np.float64), self.sr, self.tar_fs)

        score = pesq(gt_wav, gen_wav, self.tar_fs)
        return score

# for DNSMOS  ======================================================
class DNSMOS:
    def __init__(self, primary_model_path, p808_model_path, num_threads=8) -> None:
        sess_opt = ort.SessionOptions()
        sess_opt.intra_op_num_threads = num_threads
        sess_opt.inter_op_num_threads = num_threads
        self.onnx_sess = ort.InferenceSession(
                primary_model_path,
                providers=['CPUExecutionProvider'],
                sess_options=sess_opt)
        self.p808_onnx_sess = ort.InferenceSession(
                p808_model_path,
                providers=['CPUExecutionProvider'],
                sess_options=sess_opt)

    def audio_melspec(self, audio, n_mels=120, frame_size=320, hop_length=160, sr=16000, to_db=True):
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=frame_size+1, hop_length=hop_length, n_mels=n_mels)
        if to_db:
            mel_spec = (librosa.power_to_db(mel_spec, ref=np.max)+40)/40
        return mel_spec.T

    def get_polyfit_val(self, sig, bak, ovr, is_personalized_MOS):
        if is_personalized_MOS:
            p_ovr = np.poly1d([-0.00533021,  0.005101  ,  1.18058466, -0.11236046])
            p_sig = np.poly1d([-0.01019296,  0.02751166,  1.19576786, -0.24348726])
            p_bak = np.poly1d([-0.04976499,  0.44276479, -0.1644611 ,  0.96883132])
        else:
            p_ovr = np.poly1d([-0.06766283,  1.11546468,  0.04602535])
            p_sig = np.poly1d([-0.08397278,  1.22083953,  0.0052439 ])
            p_bak = np.poly1d([-0.13166888,  1.60915514, -0.39604546])

        sig_poly = p_sig(sig)
        bak_poly = p_bak(bak)
        ovr_poly = p_ovr(ovr)

        return sig_poly, bak_poly, ovr_poly

    def __call__(self, fpath, sampling_rate, is_personalized_MOS, input_length=9.01):
        aud, input_fs = sf.read(fpath)
        fs = sampling_rate
        if input_fs != fs:
            audio = librosa.resample(aud, orig_sr=input_fs, target_sr=fs)
        else:
            audio = aud
        actual_audio_len = len(audio)
        len_samples = int(input_length*fs)
        while len(audio) < len_samples:
            audio = np.append(audio, audio)

        num_hops = int(np.floor(len(audio)/fs) - input_length)+1
        hop_len_samples = fs
        predicted_mos_sig_seg_raw = []
        predicted_mos_bak_seg_raw = []
        predicted_mos_ovr_seg_raw = []
        predicted_mos_sig_seg = []
        predicted_mos_bak_seg = []
        predicted_mos_ovr_seg = []
        predicted_p808_mos = []

        for idx in range(num_hops):
            audio_seg = audio[int(idx*hop_len_samples) : int((idx+input_length)*hop_len_samples)]
            if len(audio_seg) < len_samples:
                continue

            input_features = np.array(audio_seg).astype('float32')[np.newaxis,:]
            p808_input_features = np.array(self.audio_melspec(audio=audio_seg[:-160])).astype('float32')[np.newaxis, :, :]
            oi = {'input_1': input_features}
            p808_oi = {'input_1': p808_input_features}
            p808_mos = self.p808_onnx_sess.run(None, p808_oi)[0][0][0]
            mos_sig_raw,mos_bak_raw,mos_ovr_raw = self.onnx_sess.run(None, oi)[0][0]
            mos_sig,mos_bak,mos_ovr = self.get_polyfit_val(mos_sig_raw,mos_bak_raw,mos_ovr_raw,is_personalized_MOS)
            predicted_mos_sig_seg_raw.append(mos_sig_raw)
            predicted_mos_bak_seg_raw.append(mos_bak_raw)
            predicted_mos_ovr_seg_raw.append(mos_ovr_raw)
            predicted_mos_sig_seg.append(mos_sig)
            predicted_mos_bak_seg.append(mos_bak)
            predicted_mos_ovr_seg.append(mos_ovr)
            predicted_p808_mos.append(p808_mos)

        clip_dict = {'filename': fpath, 'len_in_sec': actual_audio_len/fs, 'sr':fs}
        clip_dict['num_hops'] = num_hops
        clip_dict['OVRL_raw'] = np.mean(predicted_mos_ovr_seg_raw)
        clip_dict['SIG_raw'] = np.mean(predicted_mos_sig_seg_raw)
        clip_dict['BAK_raw'] = np.mean(predicted_mos_bak_seg_raw)
        clip_dict['OVRL'] = np.mean(predicted_mos_ovr_seg)
        clip_dict['SIG'] = np.mean(predicted_mos_sig_seg)
        clip_dict['BAK'] = np.mean(predicted_mos_bak_seg)
        clip_dict['P808_MOS'] = np.mean(predicted_p808_mos)
        return clip_dict
