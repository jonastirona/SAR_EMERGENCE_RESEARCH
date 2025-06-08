#!/bin/bash

#SBATCH --job-name=train_lstm
#SBATCH --partition=gpu
#SBATCH --account=mx6
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --time=08:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jst26@njit.edu

module load wulver
source sar-env/bin/activate

cd /mmfs1/project/mx6/jst26/SAR_EMERGENCE_RESEARCH/lstm/
python -u train_w_stats.py 12 4 110 5 140 500 0.0012 0.115