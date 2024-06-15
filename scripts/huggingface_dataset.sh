#!/bin/bash

#SBATCH --job-name=preprocess-pose-data
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --output=huggingface_dataset.out

#SBATCH --ntasks=1

set -e # exit on error
set -x # echo commands

module load anaconda3
source activate vq

# Convert to huggingface dataset
HF_DATASET_DIR="/scratch/$(whoami)/poses/huggingface"
mkdir -p $HF_DATASET_DIR

cd ..

[ ! -f "$HF_DATASET_DIR/dataset_dict.json" ] && \
python -m sign_vq.data.huggingface_dataset \
  --directory="/scratch/amoryo/poses/sign-mt-poses" \
  --output-path="$HF_DATASET_DIR"
