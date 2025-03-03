#!/bin/bash
# > bash weval.sh "LJ_V12" "gradtfktts"
# Check if required arguments are provided
if [ $# -ne 2 ]; then
    echo "Usage: bash $0 <hifigan_version> <model_name>"
	echo "hifigan_version like: LJ_V13"
	echo "model_name like:\ngradtts, gradseptts, gradtfktts, gradtfk5tts, gradtimektts, gradfreqktts, gradtfkful_plus, gradtfkful_mask"
	echo "> bash weval.sh '<exp_version>' '<model_name>'"
    exit 1
fi

# Capture arguments
HIFIGAN_VERSION=$1
MODEL_NAME=$2
BASE_DIR="result4eval/infer4colb"
#PATH_BASE_DIR="${BASE_DIR}/${MODEL_NAME}/cpu/e500_n50"

declare -A MODEL_PATHS=(
    ["gradtts"]="gradtts/cpu/e500_n50"
    ["gradseptts"]="gradseptts/cpu/e500_n50"
    ["gradtfktts"]="gradtfktts/cpu/e500_n50"
    ["gradtfk5tts"]="gradtfk5tts/cpu/e500_n50"
    ["gradtimektts"]="gradtimektts/cpu/e500_n50"
    ["gradfreqktts"]="gradfreqktts/cpu/e500_n50"
    ["gradtfkful_plus"]="gradtfkful_plus/cpu/e500_n50"
    ["gradtfkful_mask"]="gradtfkful_mask/cpu/e500_n50"
	["nix_stoch"]="nix_stoch/cpu/e500_n50"
    ["nix_deter"]="nix_deter/cpu/e500_n50"
)

# Check if model exists in mapping
if [[ -z "${MODEL_PATHS[$MODEL_NAME]}" ]]; then
    echo "Invalid model name. Available models are:"
    printf '%s\n' "${!MODEL_PATHS[@]}"
    exit 1
fi

# Construct full path
FULL_PATH="./${BASE_DIR}/${MODEL_PATHS[$MODEL_NAME]}"


# List of Python scripts to execute
SCRIPTS=(
    "wwer.py"
    "wmcd.py"
    "wlogf0rmse.py"
    "wpesq.py"
    "wdnsmos.py"
)



# Execute each script with the provided arguments
for script in "${SCRIPTS[@]}"; do
    echo "Executing $script with version $HIFIGAN_VERSION and path $FULL_PATH"
    python3 "$script" -v "$HIFIGAN_VERSION" -p "$FULL_PATH"
done
