#!/bin/bash -l
#SBATCH --job-name=mppe_prep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --account=p200539
#SBATCH --qos=default
#SBATCH --export=ALL

conda activate mppe
srun python scripts/preprocess.py --data_dir /home/users/u101975/p200539/data/pre/02/all --out data/all_tokens.json