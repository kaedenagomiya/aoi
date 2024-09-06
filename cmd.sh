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
#   intr <gpu_session>: get interactive gpu session
#   dpull: get my container_image from dockerhub
#   drun: run code with singularity
#   exec_intr: only use interactive job
#   runb: run train by batchjob
#   help: descript about usage this script


lenv_URL="docker://nagomiya/env_aoi:latest"
lenv_NAME="env_aoi_latest.sif"
DIR_ATELIER="/work/sora-sa/aoi"
INTR_SH="${DIR_ATELIER}/test_intr.sh"
RUN_SH_GT="${DIR_ATELIER}/run_train_gt.sh"
RUN_SH_SEPGT="${DIR_ATELIER}/run_train_sepgt.sh"
RUN_SH_tfk="${DIR_ATELIER}/run_train_tfk.sh"

gpu_num=1
cpu_num=16
session_time_intr="0-02:00:00"



declare -a gpu_list=(
    "gpu_intr"
    "cloudgpu1_intr"
    "ocgpu8a100_intr"
    "azuregpu1_intr"
)


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

    if [ -z ${1} ]; then
        session_mode="gpu_intr"
    elif [ ! -z ${1} ]; then
        if printf '%s\n' "${gpu_list[@]}" | grep -qx "${1}"; then
            session_mode=${1}
        else
            echo "GPU:${1} do not exist."
            exit 1
        fi
    else
        echo "Do not supported option."
        exit 1
    fi

    echo ${session_mode}
    srun -p ${session_mode} --gres=gpu:${gpu_num} -c ${cpu_num} --time=${session_time_intr} --pty bash -l
}

function cintr(){
	csession_mode="cluster_intr"
	srun -p ${csession_mode} --pty bash
	#export PATH=${PATH}:${HOME}/.local/bin
	#pip3 install --user notebook
	#jupyter-notebook --ip=0.0.0.0 --port 8888 --no-browser
}

function notebook(){
	# ./cmd.sh cintr
	# ./cmd.sh drun
	#poetry shell
	jupyter-notebook --ip=0.0.0.0 --port 8888 --no-browser
	# and add <hostname>.naist.jp
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

function exec_intr(){
    # for interractive job
    #poetry run python3 ${DIR_ATELIER}/train.py
    module load singularity cuda/12.2u2
    echo "exec by intr"
    . ${INTR_SH}
}

function run(){
    module load singularity cuda/12.2u2

    if [ -z ${1} ]; then
        echo "You need to specify option."
        exit 1
    elif [ ! -z ${1} ]; then
        if [ ${1} == "gt" ]; then
            RUN_SH=${RUN_SH_GT}
        elif [ ${1} == "sepgt" ]; then
            RUN_SH=${RUN_SH_SEPGT}
        elif [ ${1} == "tfk" ]; then
            RUN_SH=${RUN_SH_tfk}
        else
            echo "Do not supported option"
            exit 1
        fi
    else
        echo "Do not supported option."
        exit 1
    fi
    
    echo "run sequence"
    echo "${RUN_SH}"
    sbatch ${RUN_SH}
}

if [ "${1}" = "AOI" ]; then
    AOI
elif [ "${1}" = "cintr" ];then
	cintr
elif [ "${1}" = "intr" ]; then
    init_GPU ${2}
elif [ "${1}" = "dpull" ]; then
    dpull
elif [ "${1}" = "drun" ]; then
    drun
elif [ "${1}" = "exec_intr" ]; then
    exec_intr
elif [ "${1}" = "runb" ]; then
    module load singularity cuda/12.2u2
    run ${2}
elif [ "${1}" = "notebook" ]; then
	notebook
elif [ "${1}" = "help" ] || [ -z ${1} ]; then
    help
else
    echo "Do not supported this command!!"
    exit 1
fi

echo 'fin'
