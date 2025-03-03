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
# 		session: default-> gpu_intr
# 	
# 	cintr: get intractive cpu cluster session
#	runb <model_name>: run train by batchjob
# 		option:
# 			<model_name>:
#   dpull: get my container_image from dockerhub
#   drun: run container_environment with singularity
#   exec_intr: only use interactive job
# 	notebook: run jupyternotebook server
#   cap_dir: check directory size
# 		exclude: .venv, .git, wandb
#   help: descript about usage this script
# Usage:
# 	- To run environment:
# 		```
#		$ ./cmd.sh <intr or cintr>
# 		$ ./cmd.sh drun
# 		$ poetry shell
# 		if you want to run jupyternotebook server
# 		$ ./cmd.sh notebook
#		```
# 	- To run batch train:
# 		```
# 		$ ./cmd.sh runb <model_name>
#		```



DIR_ATELIER="/work/nagomiya/aoi"
#lenv_URL="docker://nagomiya/env_aoi:latest"
#lenv_NAME="env_aoi_latest.sif"
lenv_URL="docker://nagomiya/env_aoi_v2:latest"
lenv_NAME="env_aoi_v2_latest.sif"

INTR_SH="${DIR_ATELIER}/test_intr.sh"
RUN_SH_GT="${DIR_ATELIER}/run_train_gt.sh"
RUN_SH_SEPGT="${DIR_ATELIER}/run_train_sepgt.sh"
RUN_SH_tfk="${DIR_ATELIER}/run_train_tfk.sh"
RUN_SH_tfk5="${DIR_ATELIER}/run_train_tfk5.sh"
RUN_SH_timek="${DIR_ATELIER}/run_train_timek.sh"
RUN_SH_freqk="${DIR_ATELIER}/run_train_freqk.sh"
RUN_SH_tfkful="${DIR_ATELIER}/run_train_tfkful.sh"
RUN_SH_tfkfast="${DIR_ATELIER}/run_train_tfkfast.sh"

gpu_num=1
cpu_num=4
session_time_intr="0-04:00:00"

module load singularity

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

function cap_dir(){
	du -h --exclude=.venv --exclude=wandb --exclude=.git
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
	srun -p ${csession_mode} -c ${cpu_num} --pty bash
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
        elif [ ${1} == "tfk5" ]; then
            RUN_SH=${RUN_SH_tfk5}
        elif [ ${1} == "tfkful" ]; then
            RUN_SH=${RUN_SH_tfkful}
        elif [ ${1} == "timek" ]; then
            RUN_SH=${RUN_SH_timek}
        elif [ ${1} == "freqk" ]; then
            RUN_SH=${RUN_SH_freqk}
        elif [ ${1} == "tfkfast" ]; then
            RUN_SH=${RUN_SH_tfkfast}
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
elif [ "${1}" = "cap_dir" ]; then
	cap_dir
elif [ "${1}" = "help" ] || [ -z ${1} ]; then
    help
else
    echo "Do not supported this command!!"
    exit 1
fi

echo 'fin'
