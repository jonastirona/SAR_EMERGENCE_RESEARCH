#!/bin/bash

#SBATCH --job-name=debug_ns_test
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
cd /mmfs1/project/mx6/jst26/SAR_EMERGENCE_RESEARCH/ns_transformer/

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/mmfs1/project/mx6/jst26/SAR_EMERGENCE_RESEARCH:$PYTHONPATH

# Print job information
echo "=== NSTransformer Debug Test ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Working Directory: $(pwd)"
echo "Start Time: $(date)"
echo "=================================="

# Run the debug test
python -u debug_test.py

echo "=================================="
echo "End Time: $(date)"
echo "Debug test completed!"
echo "==================================" 