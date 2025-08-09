#!/bin/bash

#SBATCH --job-name=save_ars_tiles
#SBATCH --partition=gpu
#SBATCH --account=mx6
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=ALL

# Load required modules and activate environment
module load wulver
source /mmfs1/project/mx6/jst26/sar-env/bin/activate

# Change to the transformer directory
cd /mmfs1/project/mx6/jst26/SAR_EMERGENCE_RESEARCH/transformer/

source /mmfs1/project/mx6/jst26/SAR_EMERGENCE_RESEARCH/.env

# Run the script to save all ARs' tile data
python -u /project/mx6/jst26/SAR_EMERGENCE_RESEARCH/transformer/save_all_training_ars_tiles.py 