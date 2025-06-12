#!/bin/bash

#SBATCH --job-name=test_transformer
#SBATCH --partition=gpu
#SBATCH --account=mx6
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=5:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jst26@njit.edu

module load wulver
source /mmfs1/project/mx6/jst26/sar-env/bin/activate

cd /mmfs1/project/mx6/jst26/SAR_EMERGENCE_RESEARCH/transformer/

python -u train_and_eval_transformer.py \
    --dropout 0.1 \
    --hidden_size 64 \
    --num_layers 1 \
    --learning_rate 0.001 \
    --warmup_ratio 0.1 \
    --ff_ratio 4.0 \
    --num_heads 4 \
    --output_dir /mmfs1/project/mx6/jst26/SAR_EMERGENCE_RESEARCH/transformer/results