#!/bin/bash
#SBATCH --job-name=norbench
#SBATCH --account=ec30
#SBATCH --time=3:00:00
#SBATCH --mem-per-cpu=7G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --partition=accel

# Load modules
module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/
module load nlpl-scikit-bundle/1.1.1-foss-2021a-Python-3.9.5
module load nlpl-transformers/4.24.0-foss-2021a-Python-3.9.5
module load nlpl-sentencepiece/0.1.96-foss-2021a-Python-3.9.5
module load nlpl-accelerate/0.13.2-foss-2021a-Python-3.9.5



MODEL=${1}  # Path to the model or its HuggingFace name
IDENTIFIER=${2}  # Identifier to save the results
TYPE=${3}  # document or sentence? (type of the task)

echo ${MODEL}
echo ${IDENTIFIER}
echo ${TYPE}

for SEED in 10 20 30 40 50
do
  echo ${SEED}
  # --maxl 256 if out of memory
  python3 t5_sa_classification.py  -m ${MODEL} -i ${IDENTIFIER} --type ${TYPE} -d ../sentiment_analysis/${TYPE}/train.csv.gz -dev ../sentiment_analysis/${TYPE}/dev.csv.gz -t ../sentiment_analysis/${TYPE}/test.csv.gz -s ${SEED}
done

python3 analysis.py scores/${IDENTIFIER}_${TYPE}.tsv
echo scores/${IDENTIFIER}_${TYPE}.tsv
