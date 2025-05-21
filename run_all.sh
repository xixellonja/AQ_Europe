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
python ./src/seed=42/v_air_quality_for_europe_B=100_r=42_k=5.py
python ./src/seed=42/v_air_quality_for_europe_B=100_r=42.py
python ./src/seed=42/v_air_quality_for_europe_B=100_r=42_3cat.py

python ./src/seed=123/v_air_quality_for_europe_B=100_r=123_k=5.py
python ./src/seed=123/v_air_quality_for_europe_B=100_r=123.py
python ./src/seed=123/v_air_quality_for_europe_B=100_r=123_3cat.py

python ./src/seed=999/v_air_quality_for_europe_B=100_r=999_k=5.py
python ./src/seed=999/v_air_quality_for_europe_B=100_r=999.py
python ./src/seed=999/v_air_quality_for_europe_B=100_r=999_3cat.py


python ./src/seed=1616/v_air_quality_for_europe_B=100_r=1616_k=5.py
python ./src/seed=1616/v_air_quality_for_europe_B=100_r=1616.py
python ./src/seed=1616/v_air_quality_for_europe_B=100_r=1616_3cat.py


python ./src/seed=42/v_air_quality_for_europe_B=100_r=42_badge.py

# sbatch --output="./log/log_all.out" --error="./log/log_all.err" run_all.sh