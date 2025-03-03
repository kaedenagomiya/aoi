import os
import json
import yaml
import sys
import time
import copy
import IPython.display as ipd
import pprint
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import torchaudio
from librosa.filters import mel as librosa_mel_fn
#import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

import toybox

#ckpt_file_dir: logs4model/<model_name>/<runtime_name>/ckpt/
#run_tfkfulmask_k3,run_tfkfulplus_k3 by gradtfkfultts
model_name = 'gradtfk5tts' #  gradseptts, gradtfktts, gradtfk5tts, gradtimektts, gradfreqktts, gradtfkful_mask, gradtfkful_plus
model_dir = 'run_tfk_k5'
runtime_name = 'infer4colb'
hifigan_versions = 'LJ_V12'
significant_digits = 5
DEVICE='cpu'
config_path4model = 'configs/config_exp_mid.yaml'
wav_dir_name = f'wav_{hifigan_versions}'
test_ds_path = Path('configs/test_dataset.json') #Path(config['test_datalist_path'])
#~/aoi/logs4model/gradtfk5tts/run_tfk_k5/ckpt/gradtfk5tts_500_397001.pt
ckpt_filename= 'gradtfk5tts_500_397001.pt'
# for runtime to load model
print(f'runtime_name: {runtime_name}')
ckpt_dir = f'logs4model/{model_name}/{model_dir}/ckpt'
ckpt_path = Path(ckpt_dir + "/" + ckpt_filename)

if ckpt_path.exists()==True:
    print(f"ckpt_path: {ckpt_path}")
else:
    print(f"Not find")

RESULT_DIR_PATH = Path(f'./result4eval/{runtime_name}/{model_name}/{DEVICE}/e500_n50')
RESULT_MEL_DIR_PATH = RESULT_DIR_PATH / 'mel_{hifigan_versions}'
RESULT_WAV_DIR_PATH = RESULT_DIR_PATH / wav_dir_name
#RESULT_JSON_PATH = RESULT_DIR_PATH / f'eval4mid_{hifigan_versions}.json'
#eval_jsonl_path = RESULT_DIR_PATH / 'eval4mid.json'
RESULT_JSON_PATH = RESULT_DIR_PATH / f'eval4mid_{hifigan_versions}.json'

print(RESULT_DIR_PATH)
print(RESULT_MEL_DIR_PATH)
print(RESULT_WAV_DIR_PATH)

# for text2mel
print('test_ds_path-----------------------------------------')
if test_ds_path.exists():
    print(f'Exists {str(test_ds_path)}')
    with open(test_ds_path) as j:
        test_ds_list = json.load(j)
    print(f'loaded {test_ds_path}')
else:
    print(f'No exist {test_ds_path}')

config = toybox.load_yaml_and_expand_var('configs/config_exp_mid.yaml')
print(model_name)
print(runtime_name)
n_mels: int = config['n_mels'] # 80
n_fft: int = config['n_fft'] # 1024
sample_rate: int = config['sample_rate'] # 22050
hop_size: int = config['hop_size'] # 256
win_size: int = config['win_size'] # 1024
f_min: int = config['f_min'] # 0
f_max: int = config['f_max'] # 8000
random_seed: int = 1234#config['random_seed'] # 42 or 1234
print(n_mels, n_fft, sample_rate, hop_size, win_size, f_min, f_max, random_seed)

import os

print(f"all cpu at using device: {os.cpu_count()}")
print(f"Number of available CPU: {len(os.sched_getaffinity(0))}") # Number of available CPUs can also be obtained. ,use systemcall at linux.
#print(f"GPU_name: {torch.cuda.get_device_name()}\nGPU avail: {torch.cuda.is_available()}\n")

device = torch.device(DEVICE)
print(f'device: {device}')
toybox.set_seed(random_seed)

print("loading UTMOS ===================================")
predictor_utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)

infer_data_num: int = 101 #len(test_ds_list) 

eval_list = []

for i in tqdm(range(infer_data_num)):
    test_ds_filename = test_ds_list[i]['name']
    synth_wav_path = RESULT_WAV_DIR_PATH / f"{test_ds_filename}.wav"
    print(f'test_ds_index_{i}: {test_ds_filename}')
    # [seq]wav2utmos =========================================================
    print('[seq]wav2utmos')
    #iwav_path = RESULT_WAV_DIR_PATH / f"{filename}.wav"
    #wav, samplerate = torchaudio.load(iwav_path)
    wav, samplerate = torchaudio.load(synth_wav_path)
    score_utmos = predictor_utmos(wav, samplerate)
    score_utmos_float = score_utmos.item()
    print(f'utmos: {score_utmos_float}')
    #eval_dict = {'name': filename, 'path': str(iwav_path), 'utmos': score_float}
    #score_utmos_list.append(eval_dict)
    eval_list.append(score_utmos_float)


utmos_nparr = np.array(eval_list[1:101])

print(f'utmos ---------------------------')
utmos_mean = toybox.round_significant_digits(np.mean(utmos_nparr), significant_digits=significant_digits)
utmos_var = toybox.round_significant_digits(np.var(utmos_nparr), significant_digits=significant_digits)
utmos_std = toybox.round_significant_digits(np.std(utmos_nparr), significant_digits=significant_digits)
print(f'utmos mean: {utmos_mean}')
print(f'utmos var: {utmos_var}')
print(f'utmos std: {utmos_std}')



