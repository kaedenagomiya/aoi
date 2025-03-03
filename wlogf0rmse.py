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

parser = argparse.ArgumentParser(description='calc logf0rmse')

parser.add_argument('-v', '--version', required=True)
parser.add_argument('-p', '--path_base_dir', required=True)
#parser.add_argument('')

args = parser.parse_args()

hifigan_flag=str(args.version) #LJ_V*
path_base_dir=Path(args.path_base_dir)

samplerate = 16000
metrics = weval_metrix.LogF0RMSE(sr=samplerate)

seed=1234
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#ex.
#ref = toybox.load_wavtonp("data/ljspeech/LJSpeech-1.1/wavs/LJ017-0027.wav")
#synth = toybox.load_wavtonp("result4eval/infer4colb/gradtts/cpu/e500_n50/wav/LJ017-0027.wav")
    
#hifigan_flag='LJ_V14'
#path_base_dir = Path('result4eval/infer4colb/gradtts/cpu/e500_n50')
#path_base_dir = Path('result4eval/infer4colb/gradseptts/cpu/e500_n50')
#path_base_dir = Path('result4eval/infer4colb/gradtfktts/cpu/e500_n50')
#path_base_dir = Path('result4eval/infer4colb/gradtfk5tts/cpu/e500_n50')
#path_base_dir = Path('result4eval/infer4colb/gradtimektts/cpu/e500_n50')
#path_base_dir = Path('result4eval/infer4colb/gradfreqktts/cpu/e500_n50')
#path_base_dir = Path('result4eval/infer4colb/gradtfkful_plus/cpu/e500_n50')
#path_base_dir = Path('result4eval/infer4colb/gradtfkful_mask/cpu/e500_n50')

if hifigan_flag==None:
    path_wav_dir = path_base_dir / 'wav'
    RESULT_JSON_PATH = path_base_dir / 'eval4logf0rmse.json'
else:
    path_wav_dir = path_base_dir / f'wav_{hifigan_flag}'
    RESULT_JSON_PATH = path_base_dir / f'eval4logf0rmse_{hifigan_flag}.json'

#elif hifigan_flag=='LJ_V1':
#    path_wav_dir = path_base_dir / 'wav_LJ_V1'
#    RESULT_JSON_PATH = path_base_dir / 'eval4logf0rmse_LJ_V1.json'
#else:
#    print('dont supported')


# for references
ref_wav_dir = Path('data/ljspeech/LJSpeech-1.1/wavs/')
path_test_datalist = Path('./configs/test_dataset.json')
path_phn2id=Path('./configs/phn2id.json')

with open(path_test_datalist) as j:
    test_ds_list = json.load(j)


print(path_wav_dir)
print(RESULT_JSON_PATH)
print(test_ds_list[1])


# infer
print('[seq]calc score')

count = 1
start_id = 1
max_range = 101#101
eval_logf0rmse_list = []


for i in range(start_id,max_range):
    wav_name = test_ds_list[i]['name']
    path_ref_wav = ref_wav_dir / (wav_name+".wav")
    path_synth_wav = path_wav_dir / (wav_name+".wav")
    print(path_ref_wav)
    print(path_synth_wav)
    
    ref = toybox.load_wavtonp(path_ref_wav)
    synth = toybox.load_wavtonp(path_synth_wav)

    score = metrics.score(ref, synth)
    print(f'{i}: {score}')

    eval_logf0rmse_dict = {'name': wav_name, 
                        'path': str(path_synth_wav),
                        'logf0rmse': score,
                        }
    eval_logf0rmse_list.append(eval_logf0rmse_dict)
    count += 1



if RESULT_JSON_PATH.exists() == False:
    with open(RESULT_JSON_PATH, 'w') as f:
        for entry in eval_logf0rmse_list:
            f.write(json.dumps(entry) + '\n')
    print(f'Make {RESULT_JSON_PATH}')
else:
    print(f'Already Exists {RESULT_JSON_PATH}')


# get list-data for mcd
logf0rmse_values = [entry['logf0rmse'] for entry in eval_logf0rmse_list]
# 
logf0rmse_mean = np.mean(logf0rmse_values) if logf0rmse_values else 0 
logf0rmse_std = np.std(logf0rmse_values) if logf0rmse_values else 0

print(f"mean_logf0rmse: {toybox.round_significant_digits(logf0rmse_mean)}")
print(f"std_logf0rmse: {toybox.round_significant_digits(logf0rmse_std)}")

print('fin')
