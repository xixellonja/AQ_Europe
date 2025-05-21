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
python ./src/seed=42/v_air_quality_for_europe_B=100_r=42_badge.py

# sbatch --output="./log/log_42_badge.out" --error="./log/log_42_badge.err" badge_query_all_data.sh
