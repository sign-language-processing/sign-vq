#!/bin/bash

#SBATCH --job-name=sync-data
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --output=sync_bucket.out

#SBATCH --ntasks=1

set -e # exit on error
set -x # echo commands

module load anaconda3
source activate vq

poses_dir=$1
# -i: Skip copying any files that already exist at the destination, regardless of their modification time.
# -d: Delete extra files under dst_url not found under src_url. By default extra files are not deleted.
gsutil -m rsync -i -d gs://sign-mt-poses "$poses_dir"
