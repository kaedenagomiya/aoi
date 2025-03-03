#!/bin/bash

EXECDIR="/home/ldap-users-2/sora-sa/aoi"
#DOCKERENV="env_aoi_latest.sif"
#CONFIG_TRAIN="config_batch_gt_testrun.yaml"
CONFIG_TRAIN="config_tfkful_k3.yaml"

#module load singularity cuda/12.2u2
nvidia-smi

#SINGULARITYENV_CUDA_VISIBLE_DEVICE=0
CUDA_VISIBLE_DEVICE=0

#singularity -d exec --nv \
#${EXECDIR}/${DOCKERENV} \
uv run python3 ${EXECDIR}/train.py -c configs/${CONFIG_TRAIN}
