#!/bin/bash -l
#SBATCH --job-name=mppe_train1
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
srun python runner.py