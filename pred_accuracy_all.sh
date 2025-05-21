#!/usr/bin/env bash
#
#SBATCH --job-name airquality
#SBATCH --ntasks=1
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --partition=minor

# debug info
hostname
which python3
#nvidia-smi

env

# venv
source ./venv/bin/activate

#reqs 
pip install -r requirements.txt

# train
python ./src/accuracies_ALL.py
python ./src/accuracies_k=5.py
python ./src/accuracies_3cat.py

# sbatch --output="./log/log_all__pred_accuracy.out" --error="./log/log_all_pred_accuracy.err" pred_accuracy_all.sh