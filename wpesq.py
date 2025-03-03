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
import librosa
from fastdtw import fastdtw
from scipy import spatial
import weval_metrix

from pypesq import pesq as original_pesq
from pystoi import stoi
#from torchaudio.pipelines import SQUIM_OBJECTIVE, SQUIM_SUBJECTIVE
#import torchaudio.functional as F

parser = argparse.ArgumentParser(description='calc pesq, stoi, estoi')

parser.add_argument('-v', '--version', required=True)
parser.add_argument('-p', '--path_base_dir', required=True)
#parser.add_argument('')

args = parser.parse_args()

hifigan_flag=str(args.version) #LJ_V*
path_base_dir=Path(args.path_base_dir)


def si_snr(estimate, reference, epsilon=1e-8):
    estimate = estimate - estimate.mean()
    reference = reference - reference.mean()
    reference_pow = reference.pow(2).mean(axis=1, keepdim=True)
    mix_pow = (estimate * reference).mean(axis=1, keepdim=True)
    scale = mix_pow / (reference_pow + epsilon)

    reference = scale * reference
    error = estimate - reference

    reference_pow = reference.pow(2)
    error_pow = error.pow(2)

    reference_pow = reference_pow.mean(axis=1)
    error_pow = error_pow.mean(axis=1)

    si_snr = 10 * torch.log10(reference_pow) - 10 * torch.log10(error_pow)
    return si_snr.item()

# override func
def fixed_pesq(ref, deg, fs=16000, normalize=False):
    EPSILON = 1e-6
    ref = ref / np.max(np.abs(ref)) if np.any(np.abs(ref) > EPSILON) else ref
    deg = deg / np.max(np.abs(deg)) if np.any(np.abs(deg) > EPSILON) else deg
    return original_pesq(ref, deg, fs, normalize)
# if u override pypesq
#import pypesq
#pypesq.pesq = fixed_pesq


samplerate = 16000
metrics = weval_metrix.PESQ(sr=samplerate)

seed=1234
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#hifigan_flag='LJ_V14'
#ex.
#ref = toybox.load_wavtonp("data/ljspeech/LJSpeech-1.1/wavs/LJ017-0027.wav")
#synth = toybox.load_wavtonp("result4eval/infer4colb/gradtts/cpu/e500_n50/wav/LJ017-0027.wav")

#path_base_dir = Path('result4eval/infer4colb/gradtts/cpu/e500_n50')
#path_base_dir = Path('result4eval/infer4colb/gradseptts/cpu/e500_n50')
#path_base_dir = Path('result4eval/infer4colb/gradtfktts/cpu/e500_n50')
#path_base_dir = Path('result4eval/infer4colb/gradtfk5tts/cpu/e500_n50')
#path_base_dir = Path('result4eval/infer4colb/gradtimektts/cpu/e500_n50')
#path_base_dir = Path('result4eval/infer4colb/gradfreqktts/cpu/e500_n50')
#path_base_dir = Path('result4eval/infer4colb/gradtfkful_plus/cpu/e500_n50')
#path_base_dir = Path('result4eval/infer4colb/gradtfkful_mask/cpu/e500_n50')

if hifigan_flag=='':
    path_wav_dir = path_base_dir / 'wav'
    RESULT_JSON_PATH = path_base_dir / 'eval4pesq.json'
else:
    path_wav_dir = path_base_dir / f'wav_{hifigan_flag}'
    RESULT_JSON_PATH = path_base_dir / f'eval4pesq_{hifigan_flag}.json'

#elif hifigan_flag=='LJ_V1':
#    path_wav_dir = path_base_dir / 'wav_{hi}'
#    RESULT_JSON_PATH = path_base_dir / 'eval4pesq_LJ_V1.json'
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
#objective_model = SQUIM_OBJECTIVE.get_model()

count = 1
start_id = 1
max_range = 101
eval_pesq_list = []

for i in range(start_id,max_range):
    wav_name = test_ds_list[i]['name']
    path_ref_wav = ref_wav_dir / (wav_name+".wav")
    path_synth_wav = path_wav_dir / (wav_name+".wav")
    print(path_ref_wav)
    print(path_synth_wav)
    
    ref = toybox.load_wavtonp(path_ref_wav, target_sample_rate=samplerate)
    synth = toybox.load_wavtonp(path_synth_wav, target_sample_rate=samplerate)
    
    # normalize
    #ref = ref / np.max(np.abs(ref))
    #synth = synth / np.max(np.abs(synth))
    # match length
    min_len = min(len(ref), len(synth))
    ref = ref[:min_len]
    synth = synth[:min_len]
    # dtw
    #_, path = fastdtw(ref, synth, dist=spatial.distance.euclidean)
    # Extract aligned audio
    #aligned_ref = np.array([ref[i] for i, _ in path])
    #aligned_synth = np.array([synth[j] for _, j in path])
    pesq_score = metrics.score(ref, synth)
    print(pesq_score)
    stoi_score = stoi(ref, synth, samplerate, extended=False)
    estoi_score = stoi(ref, synth, samplerate, extended=True)
    print(stoi_score)
    print(estoi_score)
    eval_pesq_dict = {'name': wav_name, 
                    'path': str(path_synth_wav),
                    'pesq': pesq_score,
                    'stoi': stoi_score,
                    'estoi': estoi_score
                    }
    eval_pesq_list.append(eval_pesq_dict)
    count += 1


if RESULT_JSON_PATH.exists() == False:
    with open(RESULT_JSON_PATH, 'w') as f:
        for entry in eval_pesq_list:
            f.write(json.dumps(entry) + '\n')
    print(f'Make {RESULT_JSON_PATH}')
else:
    print(f'Already Exists {RESULT_JSON_PATH}')



# get list-data for pesq
pesq_values = [entry['pesq'] for entry in eval_pesq_list]
stoi_values = [entry['stoi'] for entry in eval_pesq_list]
estoi_values = [entry['estoi'] for entry in eval_pesq_list]
# 
pesq_mean = np.mean(pesq_values) if pesq_values else 0 
pesq_std = np.std(pesq_values) if pesq_values else 0
stoi_mean = np.mean(stoi_values) if stoi_values else 0
stoi_std = np.std(stoi_values) if stoi_values else 0
estoi_mean = np.mean(estoi_values) if estoi_values else 0
estoi_std = np.std(estoi_values) if estoi_values else 0

print(f"mean_pesq: {toybox.round_significant_digits(pesq_mean)}")
print(f"std_pesq: {toybox.round_significant_digits(pesq_std)}")

print(f"mean_stoi: {toybox.round_significant_digits(stoi_mean)}")
print(f"std_stoi: {toybox.round_significant_digits(stoi_std)}")

print(f"mean_estoi: {toybox.round_significant_digits(estoi_mean)}")
print(f"std_estoi: {toybox.round_significant_digits(estoi_std)}")

print('fin')
