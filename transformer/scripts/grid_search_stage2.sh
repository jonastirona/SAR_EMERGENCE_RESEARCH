#!/bin/bash

#SBATCH --job-name=grid_model_size
#SBATCH --partition=gpu
#SBATCH --account=mx6
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:3
#SBATCH --time=72:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jst26@njit.edu

module load wulver
source /mmfs1/project/mx6/jst26/sar-env/bin/activate

cd /mmfs1/project/mx6/jst26/SAR_EMERGENCE_RESEARCH/transformer/

# Stage 2: Model Size Grid Search
# Fixed parameters for this stage (using best values from Stage 1)
LEARNING_RATE=0.001  # Assuming this is the best value from Stage 1
WARMUP_RATIO=0.1     # Assuming this is the best value from Stage 1
FF_RATIO=4.0
NUM_HEADS=4
DROPOUT=0.1

# Calculate total combinations
total_combinations=25  # 5 hidden sizes × 5 layer counts
current_combination=0

# Grid search over hidden size and number of layers
for hidden in 32 48 64 96 128; do
    for layers in 2 3 4 5 6; do
        current_combination=$((current_combination + 1))
        echo "Running combination $current_combination of $total_combinations"
        echo "Parameters: hidden_size=$hidden, num_layers=$layers"
        echo "----------------------------------------"
        
        python -u train_and_eval_transformer.py \
            --dropout $DROPOUT \
            --hidden_size $hidden \
            --num_layers $layers \
            --learning_rate $LEARNING_RATE \
            --warmup_ratio $WARMUP_RATIO \
            --ff_ratio $FF_RATIO \
            --num_heads $NUM_HEADS \
            --output_dir /mmfs1/project/mx6/jst26/SAR_EMERGENCE_RESEARCH/transformer/results/stage2_model_size
    done
done 