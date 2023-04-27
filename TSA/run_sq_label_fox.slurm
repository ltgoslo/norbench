#!/bin/bash
#SBATCH --job-name=tsa
#SBATCH --mail-type=FAIL
#SBATCH --account=ec30
#SBATCH --partition=accel    # To use the accelerator nodes
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00     
#SBATCH --mem-per-cpu=16G
#SBATCH --nodes=1
#SBATCH --output=slurmfiles/%j.out

set -o errexit  # Recommended for easier debugging
source /etc/profile
module purge
module load PyTorch/1.11.0-foss-2021a-CUDA-11.3.1
python -m venv ~/venvs/transformers # --clear # Comment out the --clear when you already have a clean install
source ~/venvs/transformers/bin/activate
# pip install -r requirements_addon.txt --quiet  # Comment out when you already have a clean install
export TRANSFORMERS_CACHE=/cluster/work/projects/ec30/IN5550/.cache/

# d=$1
DIR=configs/*

for f in $DIR
do
    python seq_label.py $f # run_sq_label.py $f
done

