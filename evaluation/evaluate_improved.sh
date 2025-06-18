#!/bin/bash

#SBATCH --job-name=evaluate_all_trials
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
#SBATCH --mail-user=jst26@njit.edu

module load wulver
source /mmfs1/project/mx6/jst26/sar-env/bin/activate

cd /mmfs1/project/mx6/jst26/SAR_EMERGENCE_RESEARCH/evaluation/

BASE_DIR="/mmfs1/project/mx6/jst26/SAR_EMERGENCE_RESEARCH"
LSTM_PATH="${BASE_DIR}/lstm/results/t12_r4_i110_n3_h64_e1000_l0.01.pth"

# Function to parse parameters from parameters.txt
parse_parameters() {
    local params_file="$1"
    local time_window=$(grep "Time window:" "$params_file" | awk -F': ' '{print $2}')
    local num_in=$(grep "Input length:" "$params_file" | awk -F': ' '{print $2}')
    local num_layers=$(grep "Number of layers:" "$params_file" | awk -F': ' '{print $2}')
    local hidden_size=$(grep "Hidden size:" "$params_file" | awk -F': ' '{print $2}')
    local learning_rate=$(grep "Learning rate:" "$params_file" | awk -F': ' '{print $2}')
    local embed_dim=$(grep "Embedding dimension:" "$params_file" | awk -F': ' '{print $2}')
    local num_heads=$(grep "Number of heads:" "$params_file" | awk -F': ' '{print $2}')
    local ff_dim=$(grep "Feed-forward dimension:" "$params_file" | awk -F': ' '{print $2}')
    local dropout=$(grep "Dropout:" "$params_file" | awk -F': ' '{print $2}')
    
    echo "$time_window $num_in $num_layers $hidden_size $learning_rate $embed_dim $num_heads $ff_dim $dropout"
}

# Function to find model path
find_model_path() {
    local base_dir="$1"
    local trial_num="$2"
    local model_dir="${base_dir}/trial_${trial_num}/st_transformer"
    
    if [ ! -d "$model_dir" ]; then
        return 1
    fi
    
    # Find the .pth file
    local model_path=$(find "$model_dir" -name "*.pth" -type f | head -n 1)
    if [ -n "$model_path" ]; then
        echo "$model_path"
        return 0
    fi
    return 1
}

# Process random_search_3 folder (trials 16-40)
RANDOM_SEARCH_3_DIR="${BASE_DIR}/transformer/results/random_search_3"
for trial in {16..40}; do
    params_file="${RANDOM_SEARCH_3_DIR}/trial_${trial}/parameters.txt"
    if [ ! -f "$params_file" ]; then
        echo "Warning: Parameters file not found for trial $trial"
        continue
    fi
    
    model_path=$(find_model_path "$RANDOM_SEARCH_3_DIR" "$trial")
    if [ $? -ne 0 ]; then
        echo "Warning: Model file not found for trial $trial"
        continue
    fi
    
    # Parse parameters
    read -r time_window num_in num_layers hidden_size learning_rate embed_dim num_heads ff_dim dropout <<< $(parse_parameters "$params_file")
    
    # Create output directory
    output_dir="${BASE_DIR}/evaluation/results/random_search_3/trial_${trial}"
    mkdir -p "$output_dir"
    
    echo "Running trial $trial from random_search_3..."
    python -u eval_comparison_improved.py \
        --time_window "$time_window" \
        --rid_of_top 1 \
        --num_in "$num_in" \
        --num_layers "$num_layers" \
        --hidden_size "$hidden_size" \
        --learning_rate "$learning_rate" \
        --embed_dim "$embed_dim" \
        --num_heads "$num_heads" \
        --ff_dim "$ff_dim" \
        --dropout "$dropout" \
        --lstm_path "$LSTM_PATH" \
        --transformer_path "$model_path" \
        --output_dir "$output_dir"
done 