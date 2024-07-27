#!/bin/bash
#
# cmd.sh
# ----------
#
# Script to support cmd.
# Usage: ./cmd.sh
#   you need to give this shell, by chmod u+x or 755.
# Arguments:
#   AOI test command
#   intr get interactive gpu session

function help(){
    awk 'NR > 2 {
        if (/^#/) {sub("^# ?", ""); print }
        else { exit }
    }' ${0}
}

function AOI(){
    echo "Welcome to AOI system(> w <)b"
}

function intr(){
    session_mode="gpu_intr"
    gpu_num=1
    cpu_num=4
    session_time="0-01:00:00"
    module load singularity cuda/12.2u2
    srun -p ${session_mode} --gres=gpu:${gpu_num} -c ${cpu_num} --time=${session_time} --pty bash -l
}

if [ "${1}" = "AOI" ]; then
    AOI
elif [ "${1}" = "intr" ];then
    intr
elif [ -z ${1} ]; then
    help
else
    echo "Do not supported this command!!"
    exit 1
fi

echo 'fin'
