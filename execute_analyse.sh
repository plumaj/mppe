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
srun python scripts/analyse_clusters.py --models models/*.kv