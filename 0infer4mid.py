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
import argparse

import numpy as np
import torch
import torchaudio
from librosa.filters import mel as librosa_mel_fn
#import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

import toybox

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




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_index", type=int, required=True)
    parser.add_argument("-pt", "--pt_filename", type=str, required=True)
    parser.add_argument("-n", "--inferstep", type=int, required=True)
    parser.add_argument("-d", "--device", type=str, default='cpu', choices=['cpu', 'cuda', 'mps'])
    #parser.add_argument("-c", "--config", type=str, help="path to config file")
    args = parser.parse_args()
    
    model_index = args.model_index
    pt_filename = args.pt_filename
    N_STEP = args.inferstep
    TEMP = 1.5
    DEVICE = args.device

    manage_model_dict = [
        {
            "model_name": "gradtts",
            "runtime_name": "gt_k3",
        },
        {
            "model_name": "gradseptts",
            "runtime_name": "sgt_k3",
        },
        {
            "model_name": "gradtfktts",
            "runtime_name": "tfk_k3",
        },
        {
            "model_name": "gradtfk5tts",
            "runtime_name": "tfk_k5",
        },
        {
            "model_name": "gradtimektts",
            "runtime_name": "timek_k3",
        },
        {
            "model_name": "gradfreqktts",
            "runtime_name": "freqk_k3",
        },
    ]

    model_name = manage_model_dict[model_index]["model_name"]
    runtime_name = manage_model_dict[model_index]["runtime_name"]
    exp_name = 'infer4colb'
    model_epoch=500

    config_path4model = Path(f'./configs/config_{runtime_name}.yaml')
    config4model = toybox.load_yaml_and_expand_var(config_path4model)
    print(f'config_path4model: {config_path4model}')
    print(f'exists: {config_path4model.exists()}') 
    #ckpt_file_dir: logs4model/<model_name>/<runtime_name>/ckpt/
    ckpt_dir_path = Path(f'./logs4model/{model_name}/run_{runtime_name}/ckpt')
    #ckpt_path = ckpt_dir_path / f'{model_name}_{ckpt_name}'
    ckpt_path = ckpt_dir_path / f'{model_name}_{pt_filename}'
    # load exp_config
    config_yaml = 'configs/config_exp_mid.yaml'
    config = toybox.load_yaml_and_expand_var('configs/config_exp_mid.yaml')

    test_ds_path = Path(config['test_datalist_path'])
    RESULTS_JSON_NAME = 'eval4midb.json'
    RESULT_DIR_PATH = Path(f'./result4eval/{exp_name}/{model_name}/{DEVICE}/e{model_epoch}_n{N_STEP}') #run_{runtime_name}
    RESULT_MEL_DIR_PATH = RESULT_DIR_PATH / 'mel'
    RESULT_WAV_DIR_PATH = RESULT_DIR_PATH / 'wav'
    RESULT_JSON_PATH = RESULT_DIR_PATH / RESULTS_JSON_NAME
    print(f'RESULT_DIR_PATH: {RESULT_DIR_PATH}')
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
        

    # for audio params
    n_mels: int = config['n_mels'] # 80
    n_fft: int = config['n_fft'] # 1024
    sample_rate: int = config['sample_rate'] # 22050
    hop_size: int = config['hop_size'] # 256
    win_size: int = config['win_size'] # 1024
    f_min: int = config['f_min'] # 0
    f_max: int = config['f_max'] # 8000
    random_seed: int = config['random_seed'] # 1234
    print(n_mels, n_fft, sample_rate, hop_size, win_size, f_min, f_max, random_seed)

    # for phonemizer
    print(f"phn2id_path: {config['phn2id_path']}")
    with open(config['phn2id_path']) as f:
        phn2id = json.load(f)

    vocab_size = len(phn2id) + 1

    # for hifigan
    # setting file paths
    # from https://github.com/huawei-noah/Speech-Backbones/tree/main/Grad-TTS/hifi-gan
    # https://drive.google.com/drive/folders/1-eEYTB5Av9jNql0WGBlRoi-WH2J7bp5Y?usp=sharing
    HiFiGAN_CONFIG = './hifigan/official_pretrained/LJ_V2/config.json'
    HiFiGAN_ckpt = './hifigan/official_pretrained/LJ_V2/generator_v2'

    from hifigan import models, env

    with open(HiFiGAN_CONFIG) as f:
        hifigan_hparams = env.AttrDict(json.load(f))

    hifigan_randomseed = hifigan_hparams.seed
    print(f'hifigan_randomseed: {hifigan_randomseed}')

    import os

    print(f"all cpu at using device: {os.cpu_count()}")
    print(f"Number of available CPU: {len(os.sched_getaffinity(0))}") # Number of available CPUs can also be obtained. ,use systemcall at linux.
    #print(f"GPU_name: {torch.cuda.get_device_name()}\nGPU avail: {torch.cuda.is_available()}\n")
    device = torch.device(DEVICE)
    print(f'device: {device}')

    # setting random_seed ==============
    print(f'setting seed: {random_seed}')
    toybox.set_seed(random_seed)

    # load models ==========================================================================
    print(f'model_name: {model_name}')
    print("[seq] loading Model")
    print(f'model_step: {N_STEP}')
    print(f'model_temp: {TEMP}')

    print('loading ', ckpt_path)
    _, _, state_dict = torch.load(ckpt_path,
                                map_location=device)


    print("[seq] Initializing diffusion-TTS...")
    if model_name == "gradtts":
        from gradtts import GradTTS
        model = GradTTS.build_model(config4model, vocab_size)
    elif model_name == "gradseptts":
        from gradseptts import GradSepTTS
        model = GradSepTTS.build_model(config4model, vocab_size)
    elif model_name == "gradtfktts":
        from gradtfktts import GradTFKTTS
        model = GradTFKTTS.build_model(config4model, vocab_size)
    elif model_name == "gradtfk5tts":
        from gradtfk5tts import GradTFKTTS as GradTFK5TTS
        model = GradTFK5TTS.build_model(config4model, vocab_size)
    elif model_name == "gradtimektts":
        from gradtimektts import GradTimeKTTS
        model = GradTimeKTTS.build_model(config4model, vocab_size)
    elif model_name == "gradfreqktts":
        from gradfreqktts import GradFreqKTTS
        model = GradFreqKTTS.build_model(config4model, vocab_size)
    elif model_name == "gradtfkfultts":
        from gradtfkfultts import GradTFKFULTTS
        model = GradTFKFULTTS.build_model(config4model, vocab_size)
    else:
        raise ValueError(f"Error: '{model_name}' is not supported")

    model = model.to(device)
    model.load_state_dict(state_dict)
    print(f'Number of encoder + duration predictor parameters: {model.encoder.nparams/1e6}m')
    print(f'Number of decoder parameters: {model.decoder.nparams/1e6}m')
    print(f'Total parameters: {model.nparams/1e6}m')

    # generator ===================
    print("[seq] loading HiFiGAN")
    vocoder = models.Generator(hifigan_hparams)

    vocoder.load_state_dict(torch.load(
        HiFiGAN_ckpt, map_location=device)['generator'])
    vocoder = vocoder.eval().to(device)
    vocoder.remove_weight_norm()

    print("loading UTMOS ===================================")
    predictor_utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)

    # synth ==========================================================================
    eval_list = []
    infer_data_num: int = 101 #len(test_ds_list) is 200
    print(f'infer_data_num: {infer_data_num}')
    print(f'RESULT_DIR_PATH: {RESULT_DIR_PATH}')
    print(f'RESULT_MEL_DIR_PATH: {RESULT_MEL_DIR_PATH}')
    print(f'RESULT_WAV_DIR_PATH: {RESULT_WAV_DIR_PATH}')
    print(f'RESULT_JSON_PATH: {RESULT_JSON_PATH}')


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
    
    print("fin")
