#!/bin/bash

#SBATCH --job-name=preprocess-pose-data
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --output=extract_mean_std.out

#SBATCH --ntasks=1

set -e # exit on error
set -x # echo commands

module load anaconda3
source activate vq

python -m sign_vq.data.normalize --dir="$1"