import os
import re
from pathlib import Path
import glob
import random
import yaml
import math
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as taT
import matplotlib.pyplot as plt

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
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
    """
    usage:
        config = toybox.load_yaml_and_expand_var(config_path:str)
    """
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


# for plot

def plot_audio(audio, samplerate, title='time-domain waveform'):
    """
    usage:
        # audio is [channel, time(num_frames)] ex.torch.Size([1, 68608])
        # audio[0,:]: list of 1ch audio data
        # audio.shape[1]: int value of 1ch audio data length
        audio, sample_rate = torchaudio.load(str(iwav_path))
        %matplotlib inline
        plot_audio(audio, sample_rate)
    """
    # transform to mono
    channel = 0
    audio = audio[channel,:].view(1,-1)
    # to numpy
    audio = audio.to('cpu').detach().numpy().copy()
    time = np.linspace(0., audio.shape[1]/samplerate, audio.shape[1])

    fig, ax = plt.subplots(figsize=(12,9))

    ax.plot(time, audio[0, :])
    ax.set_title(title, fontsize=20, y=-0.12)
    ax.tick_params(direction='in')
    #ax.set_xlim(0, 3)
    ax.set_xlabel('Time')
    ax.set_ylabel('Amp')
    #ax.legend()
    plt.tight_layout()
    fig.canvas.draw()
    plt.show()
    #fig.savefig('figure.png')
    plt.close(fig)
    return fig

def plot_mel(tensors:list, titles:list[str]):
    """
    usage:
        mel = mel_process(...)
        fig_mel = plot_mel([mel_groundtruth[0], mel_prediction[0]],
                            ['groundtruth', 'inferenced(model)'])

    """
    xlim = max([t.shape[1] for t in tensors])
    fig, axs = plt.subplots(nrows=len(tensors),
                            ncols=1,
                            figsize=(12, 9),
                            constrained_layout=True)

    if len(tensors) == 1:
        axs = [axs]

    for i in range(len(tensors)):
        im = axs[i].imshow(tensors[i],
                           aspect="auto",
                           origin="lower",
                           interpolation='none')
        #plt.colorbar(im, ax=axs[i])
        fig.colorbar(im, ax=axs[i])
        axs[i].set_title(titles[i])
        axs[i].set_xlim([0, xlim])
    fig.canvas.draw()
    #plt.show()
    #plt.close()
    plt.close(fig)  # fig.close()
    return fig

# for text analysis to inference

def convert_phn_to_id(phonemes, phn2id):
    """
    phonemes: phonemes separated by ' '
    phn2id: phn2id dict
    """
    return [phn2id[x] for x in ['<bos>'] + phonemes.split(' ') + ['<eos>']]


def text2phnid(text, phn2id, language='en', add_blank=True):
    if language == 'en':
        from text import G2pEn
        word2phn = G2pEn()
        phonemes = word2phn(text)
        if add_blank:
            phonemes = ' <blank> '.join(phonemes)
        return phonemes, convert_phn_to_id(phonemes, phn2id)
    else:
        raise ValueError(
            'Language should be en (for English)!')
    

def round_significant_digits(value, significant_digits=5):
    if value == 0:
        return 0

    import math
    scale = math.floor(-math.log10(abs(value)))  # Find the first nonzero after the decimal point
    factor = 10 ** (scale + significant_digits - 1)  # Scale to hold 5 significant digits

    rounded_value = round(value * factor,1) / factor  # Adjust and round off the scale
    return rounded_value


def load_json(json_path:str):
    #eval_jsonl_path = Path(eval_info[eval_target])
    eval_jsonl_path = Path(json_path)
    eval_jsonl_list = []
    if eval_jsonl_path.exists() == True:
        print(f'Exist {eval_jsonl_path}')
        import json
        with open(eval_jsonl_path) as f:
            eval_jsonl_list = [json.loads(l) for l in f]
    else:
        print(f'No Exists {eval_jsonl_path}')

    return eval_jsonl_list


def calc_stoch(json_list, target_ind:str, significant_digits:int=5):
    """
    eval_jsonl_list = toybox.load_json(eval_jsonl_path)
    stoch_list = toybox.calc_stoch(eval_jsonl_list, 'utmos')

    """
    eval_list = [json_list[n][target] for n in range(len(json_list))]
    eval_nparr = np.array(eval_list[1:101])
    
    eval_mean = round_significant_digits(np.mean(eval_nparr), significant_digits=significant_digits)
    eval_var = round_significant_digits(np.var(eval_nparr), significant_digits=significant_digits)
    eval_std = round_significant_digits(np.std(eval_nparr), significant_digits=significant_digits)
    return {'mean': eval_mean, 'std': eval_std, 'var': eval_var}


def load_wavtonp(file_path, target_sample_rate=16000):
    """
    load wav, then change to 1ch, then resampling, convert numpy

    Args:
        file_path (str): wav_path
        target_sample_rate (int): after fs(default: 16kHz)
    
    Returns:
        np.ndarray: audiodata
    """
    # load wav
    waveform, sample_rate = torchaudio.load(file_path)
    
    # change to 1ch
    if waveform.size(0) > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # resampling
    if sample_rate != target_sample_rate:
        resampler = taT.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)
    
    # conv to numpy
    audio_numpy = waveform.squeeze(0).numpy() # .astype(np.float16)
    
    return audio_numpy
