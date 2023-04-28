#!/bin/bash
#SBATCH --job-name=norbench
#SBATCH --account=project_465000144
#SBATCH --time=15:00:00
#SBATCH --mem-per-cpu=7G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=standard-g

# Load modules
module --quiet purge
module load LUMI/22.08
module load cray-python/3.9.12.1
module load rocm/5.2.3

export NCCL_SOCKET_IFNAME=hsn
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_VERBOSE=2

export PYTHONUSERBASE='/projappl/project_465000498/.local'
export PATH=$PYTHONUSERBASE/bin:$PATH
export PYTHONPATH=$PYTHONUSERBASE/lib/python3.9/site-packages:$PYTHONPATH


MODEL=${1}  # Path to the model or its HuggingFace name
IDENTIFIER=${2}  # Identifier to save the results
TYPE=${3}  # document or sentence? (type of the task)

echo ${MODEL}
echo ${IDENTIFIER}
echo ${TYPE}

for SEED in 10 20 30 40 50
do
  echo ${SEED}
  python3 sa_classification.py  -m ${MODEL} -i ${IDENTIFIER} --type ${TYPE} -d ../sentiment_analysis/${TYPE}/train.csv.gz -dev ../sentiment_analysis/${TYPE}/dev.csv.gz -t ../sentiment_analysis/${TYPE}/test.csv.gz -s ${SEED}
done

python3 analysis.py scores/${IDENTIFIER}_${TYPE}.tsv
echo scores/${IDENTIFIER}_${TYPE}.tsv
