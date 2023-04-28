#!/bin/bash
#SBATCH --job-name=norbench
#SBATCH --account=ec30
#SBATCH --partition=accel
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=23:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=6

source ${HOME}/.bashrc

# sanity: exit on all errors and disallow unset environment variables
set -o errexit
set -o nounset

module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/
module load nlpl-transformers/4.20.1-foss-2021a-Python-3.9.5
module load nlpl-nlptools/2022.01-foss-2021a-Python-3.9.5
module load nlpl-datasets/2.3.2-foss-2021a-Python-3.9.5
module load nlpl-sentencepiece/0.1.96-foss-2021a-Python-3.9.5
module load nlpl-wandb/0.13.1-foss-2021a-Python-3.9.5

# print information (optional)
echo "submission directory: ${SUBMITDIR}"

MODEL=${1}  # Path to the model or its HuggingFace name
IDENTIFIER=${2}  # identifier to save the reslts and checkpoints with

echo ${MODEL}
echo ${IDENTIFIER}

python3 norbench_run.py --path_to_model ${MODEL} --task sentiment --task_specific_info document --model_name ${IDENTIFIER} --batch_size 24 --max_length 256 --seed 15,25,42,99,66
