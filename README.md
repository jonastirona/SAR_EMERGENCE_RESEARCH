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
├── evaluation/        # Evaluation scripts and utilities
├── data/             # Data directory (not tracked)
└── results/          # Model outputs and visualizations (not tracked)
```

## Requirements

- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- scikit-learn

## Usage

1. Prepare data in the appropriate format
2. Train models using the training scripts
3. Evaluate models using `eval_comparison.py`

## Evaluation

The evaluation produces:
- Side-by-side model comparisons
- Derivative analysis
- Emergence detection
- Performance metrics (RMSE, MAE, etc.) 