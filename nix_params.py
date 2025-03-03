"""
python3 nix_params.py
"""

from onnx_opcounter import calculate_params
import onnx
import onnxruntime as ort

import nix
from nix.models.TTS import NixTTSInference
from nix.tokenizers.tokenizer_en import NixTokenizerEN

model_dir_path = "./nix/pretrained/nix-ljspeech-stochastic-v0.1/"

encoder = onnx.load_model(model_dir_path+"encoder.onnx")
params_enc = calculate_params(encoder)

decoder = onnx.load_model(model_dir_path+"decoder.onnx")
params_dec = calculate_params(decoder)

params = params_enc + params_dec
print('Number of params:', params)
