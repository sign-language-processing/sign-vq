#!/bin/bash

#SBATCH --job-name=train-vq-vae
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --output=train.out

#SBATCH --ntasks=1
#SBATCH --gres gpu:1
#SBATCH --constraint=GPUMEM80GB

set -e # exit on error
set -x # echo commands

module load anaconda3
source activate vq

srun python -m sign_vq.train --data-path="$1" \
  --wandb-dir="/scratch/$(whoami)/wandb/sign-vq/"

