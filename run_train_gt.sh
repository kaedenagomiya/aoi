#!/bin/bash
#SBATCH -J btrain_gt_run
#SBATCH -p gpu_long
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH --time=01-12:00:00
#SBATCH --mail-user=nagomiya75328@gmail.com
#SBATCH --mail-type=ALL
#SBATCH -o /work/sora-sa/aoi/logs4run/stdout-%J.out
#SBATCH -e /work/sora-sa/aoi/logs4run/stderr-%J.err

#beforesetting --time=01-02:00:00

EXECDIR="/work/sora-sa/aoi"
DOCKERENV="env_aoi_latest.sif"
#CONFIG_TRAIN="config_batch_gt_testrun.yaml"
CONFIG_TRAIN="config_gt_k3.yaml"

module load singularity cuda/12.2u2
nvidia-smi

SINGULARITYENV_CUDA_VISIBLE_DEVICE=0
CUDA_VISIBLE_DEVICE=0

singularity -d exec --nv \
    ${EXECDIR}/${DOCKERENV} \
    poetry run python3 ${EXECDIR}/train.py -c configs/${CONFIG_TRAIN}
