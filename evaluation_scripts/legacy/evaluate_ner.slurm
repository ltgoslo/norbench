#!/bin/bash
#SBATCH --job-name=evaluate_lm_ner
#SBATCH --account=nn9851k
#SBATCH --partition=accel
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=3:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=4


source ${HOME}/.bashrc

# sanity: exit on all errors and disallow unset environment variables
set -o errexit
set -o nounset

module purge
module use -a /cluster/shared/nlpl/software/eb/etc/all
module load nlpl-scipy-ecosystem/2022.01-gomkl-2021a-Python-3.9.5
module load nlpl-transformers/4.20.1-gomkl-2021a-Python-3.9.5
module load nlpl-nlptools/2022.01-gomkl-2021a-Python-3.9.5
module load nlpl-datasets/2.3.2-gomkl-2021a-Python-3.9.5
module load nlpl-sentencepiece/0.1.96-gomkl-2021a-Python-3.9.5

# print information (optional)
echo "submission directory: ${SUBMITDIR}"

MODEL=${1}  # Path to the model or its HuggingFace name
LANG=${2}  # nob, nno or any other subdirectory in data/ner/
IDENTIFIER=${3}  # identifier to save the results and checkpoints with

echo ${MODEL}
echo ${LANG}
echo ${IDENTIFIER}

# by default, pass on any remaining command-line options
python3 ner_finetuning.py --model_name ${MODEL}  --training_language ${LANG} --run_model_name ${IDENTIFIER} --epochs 20
