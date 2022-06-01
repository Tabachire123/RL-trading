#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=rl
#SBATCH --output=%x.o%j
#SBATCH --ntasks=1
#SBATCH --time=05:00:00
#SBATCH --partition=gpu
#SBATCH --mem=80G
#SBATCH --gres=gpu:1

module purge
module load anaconda3/2021.05/gcc-9.2.0

./main.py
