#!/bin/bash

#SBATCH --output=outputs/%j.out
#SBATCH --error=outputs/%j.err

#SBATCH --partition=gpu
#SBATCH --nodes=1
##SBATCH --gres=gpu:v100-sxm2
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=8:00:00

#SBATCH --mail-type=FAIL
#SBATCH --mail-user=sortur.n@northeastern.edu

#SBATCH --exclude=c[2160,2162-2175,2176-2183],d1008,d1026

export PYTHONPATH="/home/sortur.n/xArm_Handovers/"

echo "node: $SLURM_NODELIST"

python agent.py

