#!/bin/bash

#SBATCH --job-name=evaluate_single_test
#SBATCH --partition=gpu
#SBATCH --account=mx6
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=ALL

module load wulver
source /mmfs1/project/mx6/jst26/sar-env/bin/activate

cd /mmfs1/project/mx6/jst26/SAR_EMERGENCE_RESEARCH/evaluation/

BASE_DIR="/mmfs1/project/mx6/jst26/SAR_EMERGENCE_RESEARCH"
LSTM_PATH="${BASE_DIR}/lstm/results/t12_r4_i110_n3_h64_e1000_l0.01.pth"

# Array of transformer model paths
TRANSFORMER_PATHS=(
    "/mmfs1/project/mx6/jst26/SAR_EMERGENCE_RESEARCH/transformer/results/st_transformer/t12_r4_i110_n3_h64_e200_l0.01.pth"
    "/mmfs1/project/mx6/jst26/SAR_EMERGENCE_RESEARCH/transformer/results/st_transformer/t12_r4_i110_n3_h64_e300_l0.005.pth"
    "/mmfs1/project/mx6/jst26/SAR_EMERGENCE_RESEARCH/transformer/results/st_transformer/t12_r4_i110_n3_h64_e400_l0.01.pth"
    "/mmfs1/project/mx6/jst26/SAR_EMERGENCE_RESEARCH/transformer/results/st_transformer/t12_r4_i110_n3_h64_e400_l0.005.pth"
    "/mmfs1/project/mx6/jst26/SAR_EMERGENCE_RESEARCH/transformer/results/st_transformer/t12_r4_i110_n3_h64_e1000_l0.01.pth"
)

# Create base output directory
output_base_dir="/mmfs1/project/mx6/jst26/SAR_EMERGENCE_RESEARCH/evaluation/results/test"
mkdir -p "$output_base_dir"

# Loop through each transformer model
trial_num=1
for TRANSFORMER_PATH in "${TRANSFORMER_PATHS[@]}"; do
    echo "========================================="
    echo "Processing Trial $trial_num: $TRANSFORMER_PATH"
    echo "========================================="
    
    # Extract filename without path and extension
    filename=$(basename "$TRANSFORMER_PATH" .pth)
    
    # Extract parameters from filename for learning rate and epochs only
    time_window=$(echo "$filename" | sed -n 's/.*t\([0-9]*\).*/\1/p')
    num_in=$(echo "$filename" | sed -n 's/.*i\([0-9]*\).*/\1/p')
    hidden_size=$(echo "$filename" | sed -n 's/.*h\([0-9]*\).*/\1/p')
    epochs=$(echo "$filename" | sed -n 's/.*e\([0-9]*\).*/\1/p')
    learning_rate=$(echo "$filename" | sed -n 's/.*l\([0-9.]*\).*/\1/p')
    
    # Use the actual architecture parameters determined from model inspection
    # These are consistent across all saved models
embed_dim=64
    num_layers=2     # All models have 2 layers despite n3 in filename
    num_heads=1      # Determined from parameter inference - this was the key issue!
    ff_dim=128       # From linear1 weight shape
    dropout=0.1      # Standard value used in training
    
    # Determine if this specific model has pre_mlp_norm by checking the model file
    echo "Checking model architecture for $filename..."
    has_pre_mlp_norm=$(python -c "
import sys
sys.path.insert(0, '/mmfs1/project/mx6/jst26/SAR_EMERGENCE_RESEARCH')
from evaluation.infer_transformer_params import infer_transformer_params
params = infer_transformer_params('$TRANSFORMER_PATH')
print('true' if params['has_pre_mlp_norm'] else 'false')
")
    
    echo "Model has pre_mlp_norm: $has_pre_mlp_norm"
    
    # Create trial-specific output directory
    output_dir="${output_base_dir}/trial_${trial_num}"
mkdir -p "$output_dir"

    echo "Trial $trial_num - Model: $filename"
    echo "Parameters extracted:"
    echo "  time_window: $time_window"
    echo "  num_in: $num_in"
    echo "  hidden_size: $hidden_size"
    echo "  epochs: $epochs"
    echo "  learning_rate: $learning_rate"
    echo "  embed_dim: $embed_dim"
    echo "  num_layers: $num_layers"
    echo "  num_heads: $num_heads"
    echo "  ff_dim: $ff_dim"
    echo "  dropout: $dropout"
    echo "  has_pre_mlp_norm: $has_pre_mlp_norm"
    echo "  output_dir: $output_dir"
    echo ""
    
    echo "Running evaluation..."
python -u eval_comparison_improved.py \
    --time_window "$time_window" \
    --num_in "$num_in" \
    --num_layers "$num_layers" \
    --hidden_size "$hidden_size" \
    --learning_rate "$learning_rate" \
    --embed_dim "$embed_dim" \
    --num_heads "$num_heads" \
    --ff_dim "$ff_dim" \
    --dropout "$dropout" \
        --has_pre_mlp_norm "$has_pre_mlp_norm" \
    --lstm_path "$LSTM_PATH" \
    --transformer_path "$TRANSFORMER_PATH" \
    --output_dir "$output_dir" 
    
    echo "Completed evaluation for Trial $trial_num ($filename)"
    echo ""
    
    # Increment trial number
    ((trial_num++))
done

echo "All evaluations completed!" 