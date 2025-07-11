#!/bin/bash

#SBATCH --job-name=new_transformer
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

# Run train_decay_new.py with specified parameters
echo "Starting weight decay search experiment"
echo "Parameters: t12 r4 i110 n3 h64 e1000 l0.01 + 4 heads, 0.1 dropout"
python -u train_decay_new.py 12 4 110 3 64 1000 0.01 0.1 