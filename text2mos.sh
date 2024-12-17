#!/bin/bash

#0:gradtts,1:gradseptts, 2:gradtfktts, 3:gradtfk5tts, 4:gradtimektts, 5:gradfreqktts
MODEL_INDEX=3
PT_FILENAME="500_397001.pt"
N_STEP=10

poetry run python3 0infer4mid.py \
	--model_index ${MODEL_INDEX} \
	--pt_filename ${PT_FILENAME} \
	--inferstep ${N_STEP}
