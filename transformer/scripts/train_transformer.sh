#!/bin/bash

#SBATCH --job-name=train_transformer
#SBATCH --partition=gpu
#SBATCH --account=mx6
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:3
#SBATCH --time=48:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jst26@njit.edu

module load wulver
source sar-env/bin/activate

cd /project/mx6/jst26/SAR_EMERGENCE_RESEARCH/transformer/
python train_w_stats.py 12 4 110 7 408 500 0.009810255183792306 st_transformer