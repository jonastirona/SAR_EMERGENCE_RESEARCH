#!/bin/bash

#SBATCH --job-name=create_emergence_sweep
#SBATCH --partition=gpu
#SBATCH --account=mx6
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
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

# Set up wandb environment variables
export WANDB_ENTITY="jonastirona-new-jersey-institute-of-technology"
export WANDB_PROJECT="transformer_emergence_optimization"

echo "Creating new sweep for emergence MSE optimization..."
echo "Project: transformer_emergence_optimization"
echo "Config file: sweep_emergence.yaml"

# Create the sweep
SWEEP_ID=$(wandb sweep sweep_emergence.yaml)

echo "Sweep created with ID: $SWEEP_ID"
echo ""
echo "To run the sweep agents, update the SWEEP_ID in sweep_emergence.sh and run:"
echo "sbatch sweep_emergence.sh"
echo ""
echo "Or to run a single agent for testing:"
echo "wandb agent $SWEEP_ID" 