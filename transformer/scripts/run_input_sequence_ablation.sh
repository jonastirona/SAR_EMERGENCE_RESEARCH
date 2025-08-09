#!/bin/bash

#SBATCH --job-name=ablation_study_v5
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

# Source environment variables
source /mmfs1/project/mx6/jst26/SAR_EMERGENCE_RESEARCH/.env

# Set up wandb environment variables
export WANDB_ENTITY="jonastirona-new-jersey-institute-of-technology"
export WANDB_PROJECT="sar-emergence-input-sequence-ablation-v5-reduce_lr"

echo "Starting Input Sequence Ablation Study V5..."
echo "Testing window lengths: 60, 90, 110, 116, 130"
echo "Fixed hyperparameters:"
echo "  - embed_dim: 128"
echo "  - num_heads: 4"
echo "  - ff_dim: 256"
echo "  - num_layers: 3"
echo "  - dropout: 0.0"
echo "  - use_temporal_conv: True"
echo "  - scheduler: reduce_lr"
echo "  - batch_size: 64"
echo "  - gradient_clip: 1.0"
echo "  - num_pred: 12"
echo "  - rid_of_top: 4"
echo "  - epochs: 1000"
echo "  - time_window: 12"
echo "  - enable_data_augmentation: False"
echo "  - total_ars: 61"

# Run the ablation study
python train_hyperparam_search_final_Version5.py

echo "Input Sequence Ablation Study V5 completed!" 