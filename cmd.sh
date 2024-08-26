#!/bin/bash
#
# cmd.sh
# ----------
#
# Script to support cmd.
# Usage: ./cmd.sh
#   you need to give this shell, by chmod u+x or 755.
# Arguments:
#   AOI: toy command
#   intr: get interactive gpu session
#   dpull: get my container_image from dockerhub
#   drun: run code with singularity
#   help: descript about usage this script


lenv_URL="docker://nagomiya/env_aoi:latest"
lenv_NAME="env_aoi_latest.sif"

module load singularity cuda/12.2u2 

function help(){
    awk 'NR > 2 {
        if (/^#/) {sub("^# ?", ""); print }
        else { exit }
    }' ${0}
}

function AOI(){
    echo "Welcome to AOI system(> w <)b"
}

function init_GPU(){
    module load singularity cuda/12.2u2

    gpu_num=1
    cpu_num=4
    session_time="0-01:00:00"

    if [ "${1}" = "intr" ]; then
        session_mode="gpu_intr"
    else
        echo "Do not supported option."
        exit 1
    fi

    srun -p ${session_mode} --gres=gpu:${gpu_num} -c ${cpu_num} --time=${session_time} --pty bash -l
}

function dpull(){
    echo "docker_URL: ${lenv_URL}"
    singularity pull ${lenv_URL}
}

function drun(){
    if [ ! -f ${lenv_NAME} ]; then
        dpull
    fi

    module load singularity cuda/12.2u2
    singularity run --nv ${lenv_NAME}
}


if [ "${1}" = "AOI" ]; then
    AOI
elif [ "${1}" = "intr" ]; then
    init_GPU intr
elif [ "${1}" = "dpull" ]; then
    dpull
elif [ "${1}" = "drun" ]; then
    drun
elif [ "${1}" = "help" ] || [ -z ${1} ]; then
    help
else
    echo "Do not supported this command!!"
    exit 1
fi

echo 'fin'
