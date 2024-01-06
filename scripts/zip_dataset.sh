#!/bin/bash

#SBATCH --job-name=preprocess-pose-data
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --output=zip_dataset.out

#SBATCH --ntasks=1

set -e # exit on error
set -x # echo commands

module load anaconda3
source activate vq

DATASET_ZIP="/scratch/$(whoami)/poses/normalized.zip"

cd ..

[ ! -f "$DATASET_ZIP" ] && \
python -m sign_vq.data.zip_dataset \
  --dir="/scratch/$(whoami)/poses/sign-mt-poses" \
  --out="$DATASET_ZIP"