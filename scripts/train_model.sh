#!/bin/bash

#SBATCH --job-name=train-vq-vae
#SBATCH --time=168:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=train.out

#SBATCH --ntasks=1
#SBATCH --gres gpu:1
#SBATCH --constraint=GPUMEM80GB

set -e # exit on error
set -x # echo commands

module load anaconda3
source activate vq

DATASET_PATH="/scratch/$(whoami)/poses/normalized.zip"

cd ..
srun python -m sign_vq.train --data-path="$DATASET_PATH" --wandb-dir="/scratch/$(whoami)/wandb/sign-vq/"


# conda activate vq
# cd sign-language/sign-vq/
# python -m sign_vq.train --data-zip="$DATASET_PATH"

