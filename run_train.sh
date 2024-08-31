#!/bin/bash
#SBATCH -p gpu_short
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH --time=0-0:30:00
#SBATCH --mail-user=nagomiya75328@gmail.com
#SBATCH --mail-type=ALL
#SBATCH -o /work/sora-sa/aoi/logs/%x-%J.out
#SBATCH -e /work/sora-sa/aoi/logs/stderr-%J.err

EXECDIR="/work/sora-sa/aoi"
DOCKERENV="env_aoi_latest.sif"

module load singularity cuda/12.2u2
nvidia-smi

SINGULARITYENV_CUDA_VISIBLE_DEVICE=0
CUDA_VISIBLE_DEVICE=0

singularity -d exec --nv \
    ${EXECDIR}/${DOCKERENV} \
    poetry run python3 ${EXECDIR}/train.py bash