# Solar Active Region (AR) Prediction Models

This repository contains implementations of LSTM and Transformer models for predicting solar active region emergence. The models analyze various solar data parameters to predict AR emergence and evolution.

## Models

- **LSTM Model**: 3-layer LSTM architecture for time series prediction
- **Transformer Model**: 2-layer Spatio-Temporal Transformer for sequence modeling

## Data

The models use the following input data:
- Power maps (multiple frequency bands)
- Magnetic flux measurements
- Intensity measurements

## Active Regions

The models are tested on several active regions:
- AR11698
- AR11726
- AR13165
- AR13179
- AR13183

## Project Structure

```
.
├── lstm/              # LSTM model implementation
├── transformer/       # Transformer model implementation
│   ├── scripts/      # Training and optimization scripts
│   └── results/      # Model results and outputs
├── evaluation/        # Evaluation scripts and utilities
├── data/             # Solar Active Region Directory
```

## Hyperparameter Optimization

The project includes two approaches for hyperparameter optimization:

1. **Grid Search**: A systematic 4-stage grid search that optimizes:
   - Stage 1: Learning rate and warmup ratio
   - Stage 2: Hidden size and number of layers
   - Stage 3: Feed-forward ratio and number of attention heads
   - Stage 4: Dropout rate

2. **Random Search**: A more efficient random search that explores:
   - Number of layers (2-4)
   - Hidden size (48-96)
   - Learning rate (0.001-0.01)
   - Embedding dimension (48-96)
   - Number of attention heads (4-12)
   - Feed-forward dimension (96-256)
   - Dropout rate (0.05-0.2)

## Requirements

- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- scikit-learn
- Optuna (for hyperparameter optimization)

## Usage

1. Prepare data in the appropriate format
2. Run hyperparameter optimization:
   ```bash
   # For grid search
   cd transformer/scripts
   ./grid_search_stage1.sh  # Run each stage sequentially
   
   # For random search
   ./random_search.sh
   ```
3. Train models using the optimized parameters
4. Evaluate models using `eval_comparison.py`

## Evaluation

The evaluation produces:
- Side-by-side model comparisons
- Derivative analysis
- Emergence detection
- Performance metrics
- Hyperparameter optimization results

## Results

Results from both grid search and random search are stored in the `transformer/results/` directory, organized by optimization method and stage.