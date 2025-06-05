#!/bin/bash

#SBATCH --job-name=optimize_lstm
#SBATCH --partition=gpu
#SBATCH --account=mx6
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jst26@njit.edu

module load wulver
source sar-env/bin/activate

cd /project/mx6/jst26/lstm/

export CUDA_LAUNCH_BLOCKING=1
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"

python optimize_hyperparams.py 