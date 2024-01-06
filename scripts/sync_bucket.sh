#!/bin/bash

#SBATCH --job-name=sync-data
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --output=sync_bucket.out

#SBATCH --ntasks=1

set -e # exit on error
set -x # echo commands

module load anaconda3

gsutil -m rsync gs://sign-mt-poses /scratch/amoryo/poses/sign-mt-poses
