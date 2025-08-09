#!/bin/bash

#SBATCH --job-name=create_new_sweep_conv
#SBATCH --partition=gpu
#SBATCH --account=mx6
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
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
export WANDB_PROJECT="transformer_temporal_conv"

# Create the sweep with the correct configuration
echo "Creating new sweep for temporal convolution transformer..."
sweep_id=$(wandb sweep sweep_conv.yaml)

echo "New sweep created with ID: $sweep_id"
echo ""
echo "Now update sweep_conv.sh with this new sweep ID:"
echo "SWEEP_ID=\"$sweep_id\""
echo ""
echo "Then run:"
echo "sbatch sweep_conv.sh" 