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
import whisper
from whisper_infer import transcribe_w_whisper
import jiwer

parser = argparse.ArgumentParser(description='calc WER')

parser.add_argument('-v', '--version', required=True)
parser.add_argument('-p', '--path_base_dir', required=True)
#parser.add_argument('')

args = parser.parse_args()

#hifigan_flag='LJ_V14'
hifigan_flag=str(args.version)
path_base_dir=Path(args.path_base_dir)
# for hypo
#path_base_dir = Path('result4eval/infer4colb/gradtts/cpu/e500_n50')
#path_base_dir = Path('result4eval/infer4colb/gradseptts/cpu/e500_n50')
#path_base_dir = Path('result4eval/infer4colb/gradtfktts/cpu/e500_n50')
#path_base_dir = Path('result4eval/infer4colb/gradtfk5tts/cpu/e500_n50')
#path_base_dir = Path('result4eval/infer4colb/gradtimektts/cpu/e500_n50')
#path_base_dir = Path('result4eval/infer4colb/gradfreqktts/cpu/e500_n50')
#path_base_dir = Path('result4eval/infer4colb/gradtfkful_plus/cpu/e500_n50')
#path_base_dir = Path('result4eval/infer4colb/gradtfkful_mask/cpu/e500_n50')

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
    RESULT_JSON_PATH = path_base_dir / 'eval4wer.json'
else:
    path_wav_dir = path_base_dir / f'wav_{hifigan_flag}'
    RESULT_JSON_PATH = path_base_dir / f'eval4wer_{hifigan_flag}.json'


#elif hifigan_flag=='LJ_V1':
#    path_wav_dir = path_base_dir / 'wav_LJ_V1'
#    RESULT_JSON_PATH = path_base_dir / 'eval4wer_LJ_V1.json'
#else:
#    print('dont supported')


# for references
path_test_datalist = Path('./configs/test_dataset.json')
path_phn2id=Path('./configs/phn2id.json')

with open(path_test_datalist) as j:
    test_ds_list = json.load(j)

print(path_wav_dir)
print(RESULT_JSON_PATH)
print(test_ds_list[1])

# load model
# https://github.com/openai/whisper/tree/main
# - 'tiny.en', 'tiny',
# - 'base.en', 'base',
# - 'small.en', 'small',
# - 'medium.en', 'medium',
# - 'large-v1', 'large-v2', 'large-v3', 'large', 'large-v3-turbo', 'turbo'
whisper_size = 'small.en' #'small.en' or 'medium.en'
print(f'[seq]load whisper_{whisper_size}')
model = whisper.load_model(whisper_size)

# infer
print('[seq]infer')

count = 1
start_id = 1
max_range = 101
eval_wer_list = []


for i in range(start_id,max_range):
    #print(count)
    #print(test_ds_list[94]['name'])
    reference = test_ds_list[i]['text']
    #path_wav = test_ds_list[i]['wav_path']
    wav_name = test_ds_list[i]['name']
    path_wav = path_wav_dir / (wav_name+".wav")
    #print(path_wav)
    #print(path_wav.is_file())
    waveform, samplerate = torchaudio.load(path_wav)

    if waveform.shape[1] == 0:
        raise ValueError("The input waveform is empty.")

    if samplerate != 16000:
        waveform = torchaudio.functional.resample(waveform=waveform, orig_freq=22050, new_freq=16000)

    #waveform_np = waveform.to('cpu').detach().numpy().copy()
    #waveform_np = np.frombuffer(waveform_np, np.int16).flatten().astype(np.float32) / 32768.0
    #waveform_np = whisper.pad_or_trim(waveform_np)

    #result = model.transcribe(waveform, verbose=True, temperature=0.8, language="en")
    result = transcribe_w_whisper(model, waveform, language="en")
    hypothesis = result['text']
    #hypothesis
    #print(f'[refer]: {reference}')
    #print(f'[hypo] : {hypothesis}')
    error = jiwer.wer(reference, hypothesis)
    print(f'{i}: {error}')
    print(str(path_wav))
    eval_wer_dict = {'name': wav_name, 
                     'path': str(path_wav),
                     'text_ref': reference,
                     'text_hypo': hypothesis,
                     'wer': error,
                    }
    eval_wer_list.append(eval_wer_dict)
    count += 1


if RESULT_JSON_PATH.exists() == False:
    with open(RESULT_JSON_PATH, 'w') as f:
        for entry in eval_wer_list:
            f.write(json.dumps(entry) + '\n')
    print(f'Make {RESULT_JSON_PATH}')
else:
    print(f'Already Exists {RESULT_JSON_PATH}')


# get list-data for wer
wer_values = [entry['wer'] for entry in eval_wer_list]

# 
wer_mean = np.mean(wer_values) if wer_values else 0 
wer_std = np.std(wer_values) if wer_values else 0

print(f"mean_WER: {toybox.round_significant_digits(wer_mean)}")
print(f"std_WER: {toybox.round_significant_digits(wer_std)}")


print('fin')
