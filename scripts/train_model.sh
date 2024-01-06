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
python -m sign_vq.train --data-path="$DATASET_PATH" --wandb-dir="/scratch/$(whoami)/wandb/sign-vq/"


# conda activate vq
# cd sign-language/sign-vq/
# python -m sign_vq.train --data-zip="$DATASET_ZIP"

# srun --pty -n 1 -c 2 --time=01:00:00 --gres=gpu:1 --constraint=GPUMEM80GB --mem=64G bash -l
# srun --pty -n 1 -c 2 --time=01:00:00 --gres=gpu:1  --mem=8G bash -l
# srun --pty -n 1 -c 1 --time=24:00:00 --mem=32G bash -l
# python -c "import torch; print([(i, torch.cuda.get_device_properties(i)) for i in range(torch.cuda.device_count())])"
# python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

#  Best validation chrf: 19.155629
#  sbatch train.sh /home/amoryo/sign-language/signbank-annotation/signbank-plus/data/parallel/original /shares/volk.cl.uzh/amoryo/checkpoints/sockeye/original
#  Best validation chrf: 28.054069
#  sbatch train.sh /home/amoryo/sign-language/signbank-annotation/signbank-plus/data/parallel/cleaned /shares/volk.cl.uzh/amoryo/checkpoints/sockeye/cleaned