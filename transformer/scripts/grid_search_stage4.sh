#!/bin/bash

#SBATCH --job-name=grid_dropout
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

# Stage 4: Dropout Regularization Grid Search
# Fixed parameters for this stage (using best values from previous stages)
LEARNING_RATE=0.001  # From Stage 1
WARMUP_RATIO=0.1     # From Stage 1
HIDDEN_SIZE=64       # From Stage 2
NUM_LAYERS=2         # From Stage 2
FF_RATIO=4.0         # From Stage 3
NUM_HEADS=4          # From Stage 3

# Grid search over dropout rates
for dropout in 0.05 0.10 0.15 0.20 0.30; do
    python -u train_and_eval_transformer.py \
        --dropout $dropout \
        --hidden_size $HIDDEN_SIZE \
        --num_layers $NUM_LAYERS \
        --learning_rate $LEARNING_RATE \
        --warmup_ratio $WARMUP_RATIO \
        --ff_ratio $FF_RATIO \
        --num_heads $NUM_HEADS \
        --output_dir /mmfs1/project/mx6/jst26/SAR_EMERGENCE_RESEARCH/transformer/results/stage4_dropout
done 