import os
import random
import json
from pathlib import Path
#from tqdm import tqdm

import numpy as np
import torch
import torchaudio
import whisper
from whisper_infer import transcribe_w_whisper
import jiwer

seed=1234
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# for references
path_test_datalist = Path('./configs/test_dataset.json')
path_phn2id=Path('./configs/phn2id.json')

with open(path_test_datalist) as j:
    test_ds_list = json.load(j)

print(test_ds_list[1])

# load model
# https://github.com/openai/whisper/tree/main
# - 'tiny.en', 'tiny',
# - 'base.en', 'base',
# - 'small.en', 'small',
# - 'medium.en', 'medium',
# - 'large-v1', 'large-v2', 'large-v3', 'large', 'large-v3-turbo', 'turbo'
whisper_size = 'medium.en' #'small.en' or 'medium.en'
print(f'[seq]load whisper_{whisper_size}')
model = whisper.load_model(whisper_size)

# infer
print('[seq]infer')

count = 1
max_range = 2

for i in range(0,max_range):
    print(count)
    reference = test_ds_list[i]['text']
    path_wav = test_ds_list[i]['wav_path']
    print(path_wav)

    waveform, samplerate = torchaudio.load(path_wav)
    if samplerate != 16000:
        waveform = torchaudio.functional.resample(waveform=waveform, orig_freq=22050, new_freq=16000)

    #waveform_np = waveform.to('cpu').detach().numpy().copy()
    #waveform_np = np.frombuffer(waveform_np, np.int16).flatten().astype(np.float32) / 32768.0
    #waveform_np = whisper.pad_or_trim(waveform_np)

    #result = model.transcribe(waveform, verbose=True, temperature=0.8, language="en")
    result = transcribe_w_whisper(model, waveform, language="en")
    hypothesis = result['text']
    #hypothesis
    print(f'[refer]: {reference}')
    print(f'[hypo] : {hypothesis}')
    error = jiwer.wer(reference, hypothesis)
    print(error)
    count += 1
