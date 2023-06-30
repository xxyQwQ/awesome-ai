#!/bin/bash

#SBATCH --job-name=kaldi
#SBATCH --partition=dgx2
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --output=%j.out
#SBATCH --error=%j.err

sh ./run.sh