#!/bin/bash

#SBATCH --job-name=test_ns_transformer
#SBATCH --partition=gpu
#SBATCH --account=mx6
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Load required modules and activate environment
module load wulver
source /mmfs1/project/mx6/jst26/sar-env/bin/activate

# Change to the ns_transformer directory
cd /mmfs1/project/mx6/jst26/SAR_EMERGENCE_RESEARCH/ns_transformer/ns_models/

# Run the test script with PYTHONPATH set to ns_transformer root
PYTHONPATH=.. python -u test_ns_transformer.py 