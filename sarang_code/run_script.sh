#!/bin/bash

#SBATCH --job-name=transformer_hyperparam_search
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
source /project/mx6/jst26/sar-env/bin/activate

# Change to the sarang_code directory
cd /project/mx6/jst26/SAR_EMERGENCE_RESEARCH/sarang_code/

# Source environment variables
source /project/mx6/jst26/SAR_EMERGENCE_RESEARCH/.env

# Set up wandb environment variables
export WANDB_ENTITY="your-wandb-username"
export WANDB_PROJECT="multi-scale-temporal-attention"

echo "========================================"
echo "TRANSFORMER HYPERPARAMETER SEARCH"
echo "PLOTS FOR EACH TRIAL"
echo "========================================"
echo "Data path: /project/mx6/jst26/SAR_EMERGENCE_RESEARCH/data"
echo "LSTM path: /project/mx6/jst26/SAR_EMERGENCE_RESEARCH/lstm/results/t12_r4_i110_n3_h64_e1000_l0.01.pth" 
echo "Output dir: /project/mx6/jst26/SAR_EMERGENCE_RESEARCH/sarang_code/hyperparam_results_ALL_TRIALS"
echo "Max trials: 16"
echo "Using ABSOLUTE paths"
echo "Creating detailed plots for EVERY trial"
echo "========================================"

# Create output directory
mkdir -p "/project/mx6/jst26/SAR_EMERGENCE_RESEARCH/sarang_code/hyperparam_results_ALL_TRIALS"
echo "Created output directory: /project/mx6/jst26/SAR_EMERGENCE_RESEARCH/sarang_code/hyperparam_results_ALL_TRIALS"

# Run hyperparameter search with ABSOLUTE paths
python hyperparam_search.py \
    --data_path "/project/mx6/jst26/SAR_EMERGENCE_RESEARCH/data" \
    --output_dir "/project/mx6/jst26/SAR_EMERGENCE_RESEARCH/sarang_code/hyperparam_results_ALL_TRIALS" \
    --lstm_path "/project/mx6/jst26/SAR_EMERGENCE_RESEARCH/lstm/results/t12_r4_i110_n3_h64_e1000_l0.01.pth" \
    --max_trials 16

echo "Hyperparameter search completed!"
echo "Results saved in: /project/mx6/jst26/SAR_EMERGENCE_RESEARCH/sarang_code/hyperparam_results_ALL_TRIALS"
echo "Check best_config.json for optimal parameters"

