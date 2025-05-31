#!/bin/bash

#SBATCH --job-name=train_lstm
#SBATCH --partition=gpu
#SBATCH --account=mx6
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jst26@njit.edu

module load wulver
module load gcc/11.2.0
module load Anaconda3
source conda.sh
conda activate lstm-env

cd /project/mx6/jst26/LSTM_RESOURCES/
python /LSTM_RESOURCES/AR_Emergence_Prediction_LSTM/train_w_stats.py 12 4 110 3 64 1000 0.01