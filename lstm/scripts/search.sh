#!/bin/bash

#SBATCH --requeue
#SBATCH --job-name=grid_search
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
#SBATCH --mail-user=ebd@njit.edu

module load wulver foss/2023b Python/3.11.5 CUDA/12.4
cd /mmfs1/project/mx6/ebd/SAR_EMERGENCE_RESEARCH/
source sar/bin/activate
pip -V
echo 'Done loading environment'

python -u lstm/grid_search.py
