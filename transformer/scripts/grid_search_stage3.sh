#!/bin/bash

#SBATCH --job-name=grid_transformer
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

# Stage 3: Transformer-specific Grid Search
# Fixed parameters for this stage (using best values from previous stages)
LEARNING_RATE=0.001  # From Stage 1
WARMUP_RATIO=0.1     # From Stage 1
HIDDEN_SIZE=64       # From Stage 2
NUM_LAYERS=2         # From Stage 2
DROPOUT=0.1

# Calculate total combinations
total_combinations=25  # 5 ff ratios × 5 head counts
current_combination=0

# Grid search over feed-forward ratio and number of heads
for ff in 2 4 6 8 10; do
    for heads in 2 4 8 16 32; do
        current_combination=$((current_combination + 1))
        echo "Running combination $current_combination of $total_combinations"
        echo "Parameters: ff_ratio=$ff, num_heads=$heads"
        echo "----------------------------------------"
        
        python -u train_and_eval_transformer.py \
            --dropout $DROPOUT \
            --hidden_size $HIDDEN_SIZE \
            --num_layers $NUM_LAYERS \
            --learning_rate $LEARNING_RATE \
            --warmup_ratio $WARMUP_RATIO \
            --ff_ratio $ff \
            --num_heads $heads \
            --output_dir /mmfs1/project/mx6/jst26/SAR_EMERGENCE_RESEARCH/transformer/results/stage3_transformer
    done
done 