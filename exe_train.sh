#!/bin/bash

EXECDIR="/work/sora-sa/aoi"
DOCKERENV=".sif"

#SBATCH -p gpu_long
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:00:00
#SBATCH -o /work/sora-sa/aoi/logs/exe_train_ep100.out

module load singularity cuda/12.2u2
nvidia-smi

SINGULARITYENV_CUDA_VISIBLE_DEVICE=0
singularity exec --nv \
    ${EXECDIR}/${DOCKERENV} \
    python3 ${EXECDIR}/train.py