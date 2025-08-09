#!/bin/bash

#SBATCH --job-name=sweep_agent_conv
#SBATCH --partition=gpu
#SBATCH --account=mx6
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --array=1-100%10
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=ALL

# Agent script for temporal convolution transformer sweep
# This script runs the wandb agent to execute sweep runs

# Set up environment
source /mmfs1/project/mx6/jst26/sar-env/bin/activate

# Navigate to the transformer directory
cd /mmfs1/project/mx6/jst26/SAR_EMERGENCE_RESEARCH/transformer

# Run the wandb agent
# Replace SWEEP_ID with the actual sweep ID from the sweep creation
echo "Starting wandb agent for temporal convolution sweep..."
wandb agent jonastirona-new-jersey-institute-of-technology/transformer_temporal_conv/yxik7cfn