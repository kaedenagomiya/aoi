import os
import re
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

# <ATTENTION>you need to do command like below for evaluate by w*.py if you synthesis speech data by tfkful_mask and tfkful_plus
# > mv result4eval/infer4colb/gradtfkfultts/cpu/e500_n50/* result4eval/infer4colb/gradtfkful_plus/cpu/e500_n50/
# > mv result4eval/infer4colb/gradtfkfultts/cpu/e500_n50/* result4eval/infer4colb/gradtfkful_mask/cpu/e500_n50/
#ckpt_file_dir: logs4model/<model_name>/<runtime_name>/ckpt/
# if you want to run_tfkfulmask_k3,run_tfkfulplus_k3 by gradtfkfultts
# you need to change TFKM process in gradtfkfultts/layers.py
model_name = 'gradtfk5tts' #  gradtts,gradseptts, gradtfktts, gradtfk5tts, gradtimektts, gradfreqktts, "gradtfkfultts for tfkful_mask and plus"
model_dir = 'run_tfk_k5' # run_tfkfulmask_k3 or run_tfkfulplus_k3, if you tfkful_*, gt, sgt, tfk_k3, tfk_k5, timek, freqk
#~/aoi/logs4model/gradtfk5tts/run_tfk_k5/ckpt/gradtfk5tts_500_397001.pt
ckpt_filename= f'{model_name}_500_397001.pt' #397001, plus500_397003 mask500_396308
hifigan_versions = 'LJ_V17'
runtime_name = 'infer4colb'
DEVICE='cpu'
config_path4model = 'configs/config_exp_mid.yaml'
wav_dir_name = f'wav_{hifigan_versions}'
test_ds_path = Path('configs/test_dataset.json') #Path(config['test_datalist_path'])

# for runtime to load model
print(f'runtime_name: {runtime_name}')
ckpt_dir = f'logs4model/{model_name}/{model_dir}/ckpt'
ckpt_path = Path(ckpt_dir + "/" + ckpt_filename)

if ckpt_path.exists()==True:
    print(f"ckpt_path: {ckpt_path}")
else:
    print(f"Not find")

RESULT_DIR_PATH = Path(f'./result4eval/{runtime_name}/{model_name}/{DEVICE}/e500_n50')
RESULT_MEL_DIR_PATH = RESULT_DIR_PATH / f'mel_{hifigan_versions}'
RESULT_WAV_DIR_PATH = RESULT_DIR_PATH / wav_dir_name
#RESULT_JSON_PATH = RESULT_DIR_PATH / f'eval4mid_{hifigan_versions}.json'
#eval_jsonl_path = RESULT_DIR_PATH / 'eval4mid.json'
RESULT_JSON_PATH = RESULT_DIR_PATH / f'eval4mid_{hifigan_versions}.json'

print(RESULT_DIR_PATH)
print(RESULT_MEL_DIR_PATH)
print(RESULT_WAV_DIR_PATH)

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

#devicesetting
import os

print(f"all cpu at using device: {os.cpu_count()}")
print(f"Number of available CPU: {len(os.sched_getaffinity(0))}") # Number of available CPUs can also be obtained. ,use systemcall at linux.
#print(f"GPU_name: {torch.cuda.get_device_name()}\nGPU avail: {torch.cuda.is_available()}\n")

device = torch.device(DEVICE)
print(f'device: {device}')
toybox.set_seed(random_seed)
# check path

# for text2mel
print('test_ds_path-----------------------------------------')
if test_ds_path.exists():
    print(f'Exists {str(test_ds_path)}')
    with open(config['test_datalist_path']) as j:
        test_ds_list = json.load(j)
    print(f'loaded {test_ds_path}')
else:
    print(f'No exist {test_ds_path}')

print('RESULT_DIR_PATH-------------------------------------------')
if RESULT_DIR_PATH.exists():
    print(f'Exists {RESULT_DIR_PATH}')
else:
    RESULT_DIR_PATH.mkdir(parents=True)
    print(f'No exist {RESULT_DIR_PATH}')

print('RESULT_MEL_DIR_PATH-------------------------------------------')
if RESULT_MEL_DIR_PATH.exists():
    print(f'Exists {RESULT_MEL_DIR_PATH}')
else:
    RESULT_MEL_DIR_PATH.mkdir(parents=True)
    print(f'No exist {RESULT_MEL_DIR_PATH}')

print('RESULT_WAV_DIR_PATH-------------------------------------------')
if RESULT_WAV_DIR_PATH.exists():
    print(f'Exists {RESULT_WAV_DIR_PATH}')
else:
    RESULT_WAV_DIR_PATH.mkdir(parents=True)
    print(f'No exist {RESULT_WAV_DIR_PATH}')

print('RESULT_JSON_PATH-------------------------------------------')
if RESULT_JSON_PATH.exists():
    print(f'Exists {RESULT_JSON_PATH}')
else:
    #RESULT_DIR_PATH.mkdir(parents=True)
    print(f'No exist {RESULT_JSON_PATH}')


print(f"phn2id_path: {config['phn2id_path']}")
with open(config['phn2id_path']) as f:
    phn2id = json.load(f)

vocab_size = len(phn2id) + 1




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




# import models
from gradtts import GradTTS
from gradseptts import GradSepTTS
from gradtfktts import GradTFKTTS
from gradtfk5tts import GradTFKTTS as GradTFK5TTS
from gradtimektts import GradTimeKTTS
from gradfreqktts import GradFreqKTTS
from gradtfkfultts import GradTFKFULTTS

print(model_name)
print("[seq] loading Model")

print("loading diffusion-TTS ===================================")
N_STEP = 50
TEMP = 1.5

print('loading ', ckpt_path)
_, _, state_dict = torch.load(ckpt_path,
                            map_location=device)

#with open(config_path4model) as f:
#    config = yaml.load(f, yaml.SafeLoader)
config4model = toybox.load_yaml_and_expand_var(config_path4model)

print("[seq] Initializing diffusion-TTS...")
if model_name == "gradtts":
    model = GradTTS.build_model(config4model, vocab_size)
elif model_name == "gradseptts":
    model = GradSepTTS.build_model(config4model, vocab_size)
elif model_name == "gradtfktts":
    model = GradTFKTTS.build_model(config4model, vocab_size)
elif model_name == "gradtfk5tts":
    model = GradTFK5TTS.build_model(config4model, vocab_size)
elif model_name == "gradtfkfultts":
    model = GradTFKFULTTS.build_model(config4model, vocab_size)
elif model_name == "gradtimektts":
    model = GradTimeKTTS.build_model(config4model, vocab_size)
elif model_name == "gradfreqktts":
    model = GradFreqKTTS.build_model(config4model, vocab_size)
else:
    raise ValueError(f"Error: '{model_name}' is not supported")

model = model.to(device)
model.load_state_dict(state_dict)
print(f'Number of encoder + duration predictor parameters: {model.encoder.nparams/1e6}m')
print(f'Number of decoder parameters: {model.decoder.nparams/1e6}m')
print(f'Total parameters: {model.nparams/1e6}m')

print("loading HiFi-GAN ===================================")
#setting file paths
# from https://github.com/huawei-noah/Speech-Backbones/tree/main/Grad-TTS/hifi-gan
# https://drive.google.com/drive/folders/1-eEYTB5Av9jNql0WGBlRoi-WH2J7bp5Y?usp=sharing
if re.search('LJ_V1.*', hifigan_versions) != None:
    HiFiGAN_CONFIG = f'./hifigan/official_pretrained/LJ_V1/config.json'
    HiFiGAN_ckpt = './hifigan/official_pretrained/LJ_V1/generator_v1'
    print(f"LJ_V1")
elif re.search('LJ_V2.*', hifigan_versions) != None:
    HiFiGAN_CONFIG = './hifigan/official_pretrained/LJ_V2/config.json'
    HiFiGAN_ckpt = './hifigan/official_pretrained/LJ_V2/generator_v2'
    print(f"LJ_V2")
else:
    print('Dont supported.')
    raise ValueError

print(f'hifigan_vesions: {hifigan_versions}')

from hifigan import models, env

with open(HiFiGAN_CONFIG) as f:
    hifigan_hparams = env.AttrDict(json.load(f))

hifigan_randomseed = hifigan_hparams.seed
print(f'hifigan_randomseed: {hifigan_randomseed}')

# generator ===================
print("[seq] loading HiFiGAN")
vocoder = models.Generator(hifigan_hparams)

vocoder.load_state_dict(torch.load(
    HiFiGAN_ckpt, map_location=device)['generator'])
vocoder = vocoder.eval().to(device)
vocoder.remove_weight_norm()

print("loading UTMOS ===================================")
predictor_utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)


infer_data_num: int = 101 #len(test_ds_list) is 200
print(f'infer_data_num: {infer_data_num}')
print(f'RESULT_DIR_PATH: {RESULT_DIR_PATH}')
print(f'RESULT_MEL_DIR_PATH: {RESULT_MEL_DIR_PATH}')
print(f'RESULT_WAV_DIR_PATH: {RESULT_WAV_DIR_PATH}')
print(f'RESULT_JSON_PATH: {RESULT_JSON_PATH}')


eval_list = []


for i in tqdm(range(infer_data_num)):
    test_ds_filename = test_ds_list[i]['name']
    mel_npy_path = RESULT_MEL_DIR_PATH / f"{test_ds_filename}.npy"
    synth_wav_path = RESULT_WAV_DIR_PATH / f"{test_ds_filename}.wav"
    print(f'test_ds_index_{i}: {test_ds_filename}')
    # [seq]text2mel =========================================================
    # load txt
    print('[seq]text2mel')
    text = test_ds_list[i]['text']
    phonemes, phnid = text2phnid(text, phn2id, 'en')
    phonemes_len_int = len(phonemes)
    phnid_len_int = len(phnid)
    print(f'phonemes_len: {phonemes_len_int}')
    print(f'phnid_len: {phnid_len_int}')
    phnid_len = torch.tensor(len(phnid), dtype=torch.long).unsqueeze(0).to(device)
    phnid = torch.tensor(phnid).unsqueeze(0).to(device)

    # [seq] synth speech
    # process text to mel
    # mel is [n_mels, n_frame]
    start_time = time.perf_counter()
    _, mel_prediction, _ = model.forward(phnid,
                                        phnid_len,
                                        n_timesteps=N_STEP,
                                        temperature=TEMP,
                                        solver='original')
    end_time = time.perf_counter()

    dt = end_time - start_time
    dt4mel = dt * 22050 / ( mel_prediction.shape[-1] * 256)
    print(f'{model_name} dt: {dt}')
    print(f'{model_name} RTF: {dt4mel}')

    # for save mel
    mel4save = mel_prediction.unsqueeze(0) # [batch, channel(freq), n_frame(time)] ex.[1, 80, 619]
    # save
    #mel_npy_path =  RESULT_MEL_DIR_PATH / f"{test_ds_filename}.npy"
    #print(f'test_ds_index_{i}: {mel_npy_path}')
    np.save(mel_npy_path, mel4save.cpu().detach().numpy().copy())

    # [seq]mel2wav =========================================================
    print('[seq]mel2wav')
    x = np.load(mel_npy_path) # [1, n_mel, n_frame]
    x2audio = torch.FloatTensor(x).to(device)
    x2audio = x2audio.squeeze().unsqueeze(0)
    # x2audio is [1, n_mels, n_frames]
    assert x2audio.shape[0] == 1
    with torch.no_grad():
        # vocoder.forward(x).cpu() is torch.Size([1, 1, 167168])
        audio = (vocoder.forward(x2audio).cpu().squeeze().clamp(-1,1).numpy() * 32768).astype(np.int16)
    write(
        synth_wav_path,
        hifigan_hparams.sampling_rate,
        audio)

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

    # path, テキスト文、phonimes, phonimes数, dt, RTF, utmos
    eval_dict = {
        'name': test_ds_filename,
        'phonemes_len': phonemes_len_int,
        'phnid_len': phnid_len_int,
        'dt': dt,
        'RTF4mel': dt4mel,
        'utmos': score_utmos_float
    }
    eval_list.append(eval_dict)


#RESULT_JSON_PATH = RESULT_DIR_PATH / 'eval4mid.json'
if RESULT_JSON_PATH.exists() == False:
    with open(RESULT_JSON_PATH, 'w') as f:
        for entry in eval_list:
            f.write(json.dumps(entry) + '\n')
    print(f'Make {RESULT_JSON_PATH}')
else:
    print(f'Already Exists {RESULT_JSON_PATH}')



print('fin')





