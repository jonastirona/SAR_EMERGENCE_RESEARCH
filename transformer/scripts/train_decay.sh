#!/bin/bash

#SBATCH --job-name=weight_decay_search
#SBATCH --partition=gpu
#SBATCH --account=mx6
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:3
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=ALL

# Load required modules and activate environment
module load wulver
source /mmfs1/project/mx6/jst26/sar-env/bin/activate

# Change to the transformer directory
cd /mmfs1/project/mx6/jst26/SAR_EMERGENCE_RESEARCH/transformer/

source /mmfs1/project/mx6/jst26/SAR_EMERGENCE_RESEARCH/.env

# Run weight decay search
echo "Starting weight decay search (weight_decays: 1e-3, 1e-4, 1e-5)"
python -u train_decay.py 