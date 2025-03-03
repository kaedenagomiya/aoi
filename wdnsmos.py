"""
python3 wdnsmos.py -v LJ_V1 -p <path_base_dir>
"""

import os
import random
import json
from pathlib import Path
import toybox
#from tqdm import tqdm
import argparse

import numpy as np
import torch
import torchaudio
import weval_metrix

parser = argparse.ArgumentParser(description='calc dnsmos')

parser.add_argument('-v', '--version', required=True)
parser.add_argument('-p', '--path_base_dir', required=True)

args = parser.parse_args()

hifigan_flag=str(args.version) #LJ_V*
path_base_dir=Path(args.path_base_dir)

#ex.
#ref = toybox.load_wavtonp("data/ljspeech/LJSpeech-1.1/wavs/LJ017-0027.wav")
#synth = toybox.load_wavtonp("result4eval/infer4colb/gradtts/cpu/e500_n50/wav/LJ017-0027.wav")

#hifigan_flag='LJ_V1'
#path_base_dir = Path('result4eval/infer4colb/gradtts/cpu/e500_n50')
#path_base_dir = Path('result4eval/infer4colb/gradseptts/cpu/e500_n50')
#path_base_dir = Path('result4eval/infer4colb/gradtfktts/cpu/e500_n50')
#path_base_dir = Path('result4eval/infer4colb/gradtfk5tts/cpu/e500_n50')
#path_base_dir = Path('result4eval/infer4colb/gradtimektts/cpu/e500_n50')
#path_base_dir = Path('result4eval/infer4colb/gradfreqktts/cpu/e500_n50')
#path_base_dir = Path('result4eval/infer4colb/gradtfkful_plus/cpu/e500_n50')
#path_base_dir = Path('result4eval/infer4colb/gradtfkful_mask/cpu/e500_n50')


dnsmos = weval_metrix.DNSMOS(
            "DNSMOS/DNSMOS/sig_bak_ovr.onnx",
            "DNSMOS/DNSMOS/model_v8.onnx",
            num_threads=8)


SAMPLING_RATE = 16000
INPUT_LENGTH = 9.01

#wav_path = 'data/ljspeech/LJSpeech-1.1/wavs/LJ031-0171.wav'


seed=1234
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


if hifigan_flag=='':
    path_wav_dir = path_base_dir / 'wav'
    RESULT_JSON_PATH = path_base_dir / 'eval4dnsmos.json'
else:
    path_wav_dir = path_base_dir / f'wav_{hifigan_flag}'
    RESULT_JSON_PATH = path_base_dir / f'eval4dnsmos_{hifigan_flag}.json'

# for references
#ref_wav_dir = Path('data/ljspeech/LJSpeech-1.1/wavs/')
path_test_datalist = Path('./configs/test_dataset.json')
#path_phn2id=Path('./configs/phn2id.json')

with open(path_test_datalist) as j:
    test_ds_list = json.load(j)

print(path_wav_dir)
print(RESULT_JSON_PATH)
print(test_ds_list[1])

# infer
print('[seq]calc score')

count = 1
start_id = 1
max_range = 101
eval_dnsmos_list = []

for i in range(start_id,max_range):    
    wav_name = test_ds_list[i]['name']
    path_synth_wav = path_wav_dir / (wav_name+".wav")

    #{'filename': 'data/ljspeech/LJSpeech-1.1/wavs/LJ031-0171.wav', 
    # 'len_in_sec': 1.876375, 'sr': 16000, 'num_hops': 6,
    # 'OVRL_raw': np.float32(3.4388416), 'SIG_raw': np.float32(3.78549), 'BAK_raw': np.float32(4.0435643), 
    # 'OVRL': np.float64(3.081215277177979), 'SIG': np.float64(3.4228541711019385), 'BAK': np.float64(3.9571496919321762), 
    # 'P808_MOS': np.float32(3.6507962)}
    score = dnsmos(path_synth_wav, SAMPLING_RATE, False, input_length=INPUT_LENGTH)
    # filename, len_in_sec, P808_MOS
    print(score["P808_MOS"])

    eval_dnsmos_dict = {'name': wav_name, 
                    'path': str(path_synth_wav),
                    'dnsovrl': float(score["OVRL"]),
                    'dnssig': float(score["SIG"]),
                    'dnsbak': float(score["BAK"]),
                    'dnsmos': float(score["P808_MOS"]),
                    }
    eval_dnsmos_list.append(eval_dnsmos_dict)
    count += 1


if RESULT_JSON_PATH.exists() == False:
    with open(RESULT_JSON_PATH, 'w') as f:
        for entry in eval_dnsmos_list:
            f.write(json.dumps(entry) + '\n')
    print(f'Make {RESULT_JSON_PATH}')
else:
    print(f'Already Exists {RESULT_JSON_PATH}')


# get list-data for dnsmos
dnsmos_values = [entry['dnsmos'] for entry in eval_dnsmos_list]
# 
dnsmos_mean = np.mean(dnsmos_values) if dnsmos_values else 0 
dnsmos_std = np.std(dnsmos_values) if dnsmos_values else 0

print(f"mean_dnsmos: {toybox.round_significant_digits(dnsmos_mean)}")
print(f"std_dnsmos: {toybox.round_significant_digits(dnsmos_std)}")


print('fin')
