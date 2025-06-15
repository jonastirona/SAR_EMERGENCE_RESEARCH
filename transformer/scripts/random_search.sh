#!/bin/bash

#SBATCH --job-name=random_search
#SBATCH --partition=gpu
#SBATCH --account=mx6
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --time=72:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jst26@njit.edu

# Exit on error
set -e

module load wulver
source sar-env/bin/activate

cd /project/mx6/jst26/SAR_EMERGENCE_RESEARCH/transformer/

# Create base directory for random search
BASE_DIR="results/random_search_2"
mkdir -p "$BASE_DIR"

# Number of trials to perform
N_TRIALS=15

# Fixed parameters
NUM_PRED=12        # Time window
RID_OF_TOP=4       # Rid of top
NUM_IN=110         # Input length
N_EPOCHS=400       # Number of epochs

# Function to generate random number in range
random_range() {
    local min=$1
    local max=$2
    echo $(( RANDOM % (max - min + 1) + min ))
}

# Function to generate random float in range
random_float() {
    local min=$1
    local max=$2
    echo $(awk -v min=$min -v max=$max 'BEGIN{srand(); print min+rand()*(max-min)}')
}

# Function to generate valid embed_dim based on num_heads
generate_valid_embed_dim() {
    local num_heads=$1
    local min_dim=48
    local max_dim=96
    # Find the smallest multiple of (num_heads * 2) that's >= min_dim
    local min_multiple=$(( (min_dim + (num_heads * 2) - 1) / (num_heads * 2) * (num_heads * 2) ))
    # Find the largest multiple of (num_heads * 2) that's <= max_dim
    local max_multiple=$(( max_dim / (num_heads * 2) * (num_heads * 2) ))
    # Generate a random multiple of (num_heads * 2) in this range
    local range=$(( (max_multiple - min_multiple) / (num_heads * 2) + 1 ))
    echo $(( min_multiple + (RANDOM % range) * (num_heads * 2) ))
}

# Function to run a single training with random parameters
run_training() {
    local trial_num=$1
    local trial_dir="${BASE_DIR}/trial_${trial_num}"
    mkdir -p "$trial_dir"

    # Generate random parameters for model architecture
    NUM_LAYERS=$(random_range 2 4)        # Number of layers
    HIDDEN_SIZE=$(random_range 48 96)     # Hidden size
    LEARNING_RATE=$(random_float 0.001 0.01)  # Learning rate
    NUM_HEADS=$(random_range 4 12)        # Number of heads
    EMBED_DIM=$(generate_valid_embed_dim $NUM_HEADS)  # Embedding dimension (divisible by num_heads)
    FF_DIM=$(random_range 96 256)         # Feed-forward dimension
    DROPOUT=$(random_float 0.05 0.2)      # Dropout

    # Save parameters to file
    {
        echo "Parameters for trial ${trial_num}:"
        echo "Fixed Parameters:"
        echo "Time window: ${NUM_PRED}"
        echo "Rid of top: ${RID_OF_TOP}"
        echo "Input length: ${NUM_IN}"
        echo "Number of epochs: ${N_EPOCHS}"
        echo ""
        echo "Randomized Parameters:"
        echo "Number of layers: ${NUM_LAYERS}"
        echo "Hidden size: ${HIDDEN_SIZE}"
        echo "Learning rate: ${LEARNING_RATE}"
        echo "Embedding dimension: ${EMBED_DIM}"
        echo "Number of heads: ${NUM_HEADS}"
        echo "Feed-forward dimension: ${FF_DIM}"
        echo "Dropout: ${DROPOUT}"
    } > "${trial_dir}/parameters.txt"

    # Run training
    echo "Starting trial ${trial_num} with parameters:"
    cat "${trial_dir}/parameters.txt"
    echo "----------------------------------------"

    # Run training - let SLURM handle output
    python3 train_w_stats.py \
        "${NUM_PRED}" \
        "${RID_OF_TOP}" \
        "${NUM_IN}" \
        "${NUM_LAYERS}" \
        "${HIDDEN_SIZE}" \
        "${N_EPOCHS}" \
        "${LEARNING_RATE}" \
        "${EMBED_DIM}" \
        "${NUM_HEADS}" \
        "${FF_DIM}" \
        "${DROPOUT}" \
        "${trial_dir}"

    # Check if training completed successfully
    if [ $? -ne 0 ]; then
        echo "Error: Training failed for trial ${trial_num}"
        return 1
    fi

    echo "Completed trial ${trial_num}"
    echo "----------------------------------------"
}

# Main execution
echo "Starting random search with ${N_TRIALS} trials"
echo "Results will be saved in ${BASE_DIR}"
echo "----------------------------------------"

# Run all trials
for i in $(seq 1 ${N_TRIALS}); do
    if ! run_training $i; then
        echo "Warning: Trial $i failed, continuing with next trial..."
    fi
done

echo "Random search completed. Results saved in ${BASE_DIR}"

# Create summary file
echo "Creating summary of all trials..."
{
    echo "Random Search Summary"
    echo "===================="
    echo "Date: $(date)"
    echo "Number of trials: ${N_TRIALS}"
    echo ""
    echo "Fixed Parameters:"
    echo "Time window: ${NUM_PRED}"
    echo "Rid of top: ${RID_OF_TOP}"
    echo "Input length: ${NUM_IN}"
    echo "Number of epochs: ${N_EPOCHS}"
    echo ""
    echo "Trial Parameters:"
    echo "--------------"
} > "${BASE_DIR}/summary.txt"

# Add parameters from each trial to summary
for i in $(seq 1 ${N_TRIALS}); do
    if [ -f "${BASE_DIR}/trial_${i}/parameters.txt" ]; then
        {
            echo "Trial ${i}:"
            cat "${BASE_DIR}/trial_${i}/parameters.txt"
            echo ""
        } >> "${BASE_DIR}/summary.txt"
    fi
done

echo "Summary created at ${BASE_DIR}/summary.txt" 