import os
import json
import yaml
import sys
import time
import copy
from copy import deepcopy
import pprint
from pathlib import Path
from tqdm import tqdm

import numpy as np
#import torch
#import torchaudio
#from librosa.filters import mel as librosa_mel_fn
#import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
#from scipy.io.wavfile import write
import argparse

import toybox

parser = argparse.ArgumentParser(description='load eval.json and culc static val')
parser.add_argument('-it', '--ind_target', required=True, help='eval_indicator')
parser.add_argument('-p', '--path',required=True, help='dir_path')
parser.add_argument('-mt', '--model_target', help='model_target')
args = parser.parse_args()

significant_digits = 5

target=args.ind_target
eval_jsonl_path = Path(args.path)
#model_target = args.model_target

#target = "utmos" # dt, RTF4mel, utmos, wer,
#eval_base_dir = Path( 'result4eval/infer4colb/gradtfk5tts/cpu/e500_n50/')

#if target in ['dt', 'RTF4mel', 'utmos']:
#    eval_jsonl_path = eval_base_dir / 'eval4midb.json'

#eval_jsonl_path = eval_base_dir/'eval4midb.json'
#eval_jsonl_path = eval_base_dir
eval_jsonl_list = toybox.load_json(eval_jsonl_path)


#print(eval_jsonl_list[0])

eval_list = [eval_jsonl_list[n][target] for n in range(len(eval_jsonl_list))]
eval_nparr = np.array(eval_list[1:101])

if target=="stoi" or target=="estoi":
    #eval_nparr_all = deepcopy(eval_nparr)
    #eval_nparr = eval_nparr_all[eval_nparr_all >= 0]
    #exclude = len(eval_nparr_all) - len(eval_nparr)
    #print(exclude)
    #print(len(eval_nparr))
    eval_nparr = np.where(eval_nparr < 0.0, 0.0, eval_nparr)
    print(f'num_of_zero: {len(eval_nparr) - np.count_nonzero(eval_nparr)}')

#print(f'data_len: {len(eval_nparr)}')
# for culc difference time to infer text2mel
eval_mean = toybox.round_significant_digits(np.mean(eval_nparr), significant_digits=significant_digits)
eval_var = toybox.round_significant_digits(np.var(eval_nparr), significant_digits=significant_digits)
eval_std = toybox.round_significant_digits(np.std(eval_nparr), significant_digits=significant_digits)
#print(f'{target_model}: dt ---------------------------')
print(f'{target} mean: {eval_mean}')
print(f'{target} std: {eval_std}')
print(f'{target} var: {eval_var}')
