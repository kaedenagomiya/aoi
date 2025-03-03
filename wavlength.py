?mport os
import json
import yaml
import sys
import time
import copy
from tqdm import tqdm
from pathlib import Path
import torchaudio

import toybox


test_ds_path = Path('configs/test_dataset.json') #Path(config['test_datalist_path'])
RESULT_WAV_DIR_PATH = Path('result4eval/infer4colb/gradtfk5tts/cpu/e500_n50/wav_LJ_V12') 


print('test_ds_path-----------------------------------------')
if test_ds_path.exists():
    print(f'Exists {str(test_ds_path)}')
    with open(test_ds_path) as j:
        test_ds_list = json.load(j)
    print(f'loaded {test_ds_path}')
else:
    print(f'No exist {test_ds_path}')

print('RESULT_WAV_DIR_PATH-------------------------------------------')
if RESULT_WAV_DIR_PATH.exists():
    print(f'Exists {RESULT_WAV_DIR_PATH}')
else:
    RESULT_WAV_DIR_PATH.mkdir(parents=True)
    print(f'No exist {RESULT_WAV_DIR_PATH}')



infer_data_num: int = 1 #101 #len(test_ds_list) is 200

for i in tqdm(range(infer_data_num)):
    test_ds_filename = test_ds_list[i]['name']
    #mel_npy_path = RESULT_MEL_DIR_PATH / f"{test_ds_filename}.npy"
    synth_wav_path = RESULT_WAV_DIR_PATH / f"{test_ds_filename}.wav"
    print(f'test_ds_index_{i}: {test_ds_filename}')
    wav, samplerate = torchaudio.load(synth_wav_path)
    # metadata = torchaudio.info()
    length_wav = len(wav[0])
    print(length_wav, samplerate)
    dur_time = length_wav / samplerate
    print(dur_time)



