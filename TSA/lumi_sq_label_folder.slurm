#!/bin/bash

#SBATCH --job-name=seq-label
#SBATCH --account=project_465000144
#SBATCH --time=09:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --partition=small-g
#SBATCH --output=lumi-%j.out


set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error


source ${HOME}/.bashrc

# Load modules
module --quiet purge
module load LUMI/22.08
module load cray-python/3.9.12.1
module load rocm/5.0.2

export NCCL_SOCKET_IFNAME=hsn
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_VERBOSE=2

# Scratch folder: /scratch/project_465000144/egilron/sq_label

source /project/project_465000144/pytorch_1.13.1/bin/activate
export TRANSFORMERS_CACHE=/scratch/project_465000144/egilron/.cache/
# pip install -r requirements_addon_lumi.txt
f=$1
DIR=${f}/*

for f in $DIR
do
    echo $f
    python seq_label.py $f
done