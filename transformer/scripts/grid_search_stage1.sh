#!/bin/bash

#SBATCH --job-name=grid_lr_warmup
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

# Stage 1: Learning Rate & Warmup Ratio Grid Search
# Fixed parameters for this stage
HIDDEN_SIZE=64
NUM_LAYERS=2
FF_RATIO=4.0
NUM_HEADS=4
DROPOUT=0.1

# total combinations
total_combinations=6  # 3 learning rates × 2 warmup ratios
current_combination=0

# Grid search over learning rate and warmup ratio
for lr in 0.001 0.005 0.01; do
    for warmup in 0.05 0.1; do
        current_combination=$((current_combination + 1))
        echo "Running combination $current_combination of $total_combinations"
        echo "Parameters: lr=$lr, warmup=$warmup"
        echo "----------------------------------------"
        
        python -u train_and_eval_transformer.py \
            --dropout $DROPOUT \
            --hidden_size $HIDDEN_SIZE \
            --num_layers $NUM_LAYERS \
            --learning_rate $lr \
            --warmup_ratio $warmup \
            --ff_ratio $FF_RATIO \
            --num_heads $NUM_HEADS \
            --output_dir /mmfs1/project/mx6/jst26/SAR_EMERGENCE_RESEARCH/transformer/results/stage1_lr_warmup
    done
done 


# stage 1 best results:
# learning_rate": 0.01,
# warmup_ratio": 0.05,