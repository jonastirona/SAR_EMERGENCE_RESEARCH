
# SAR Emergence Research Project

This repository contains the implementation of LSTM and Transformer models for predicting Solar Active Region (SAR) emergence patterns.

## Table of Contents
1. [Environment Setup](#environment-setup)
2. [Wulver Cluster Setup](#wulver-cluster-setup)
3. [Project Structure](#project-structure)
4. [Configuration](#configuration)
5. [Training Pipeline](#training-pipeline)
6. [Evaluation Pipeline](#evaluation-pipeline)
7. [Data Structure](#data-structure)

## Environment Setup

1. **Clone the repository** and enter the directory:
   ```bash
   git clone <the-repo-url>
   cd path_it_was_cloned_to
   ```

2. **Create and activate a Python virtual environment**:
   ```bash
   python3 -m venv sar-env
   source sar-env/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Wulver Cluster Setup

1. **Load required modules**:
   ```bash
   module load wulver
   ```

2. **Activate environment**:
   ```bash
   source /mmfs1/project/PI-ucid/your-ucid/sar-env/bin/activate
   ```

3. **Environment Configuration**:
   Create a `.env` file in the project root with:
   ```
   WANDB_API_KEY=your_wandb_api_key_here
   WANDB_ENTITY="jonastirona-new-jersey-institute-of-technology"
   WANDB_PROJECT="sar-emergence"
   ```

## Project Structure
SAR_EMERGENCE_RESEARCH/
├── data/ # Raw data directory
├── transformer/
│ ├── models/ # Model architectures
│ ├── scripts/ # Training scripts
│ ├── train.py # Main transformer training script
│ ├── train_wandb.py # Training with W&B logging
│ └── eval.py # Evaluation script
├── lstm/
│ ├── train_w_stats.py # LSTM training with statistics
│ ├── eval.py # LSTM evaluation
│ └── functions.py # Helper functions
└── README.md

## Training Pipeline

### Transformer Training

1. **`train.py`**:
   ```bash
   cd SAR_EMERGENCE_RESEARCH/transformer
   python train.py
   ```
   This will run a batch size search experiment with:
   - Fixed learning rate (0.001)
   - Fixed dropout (0.3)
   - Batch sizes: [64, 128, 256, 512]

2. **Run `train.sh`**:
   ```bash
   cd SAR_EMERGENCE_RESEARCH/transformer/scripts
   sbatch train.sh
   ```
   This will run train.py on the NJIT Wulver HPC.
### LSTM Training

1. **`train_w_stats.py`**:
   ```bash
   cd SAR_EMERGENCE_RESEARCH/lstm
   python train_w_stats.py <num_pred> <rid_of_top> <num_in> <num_layers> <hidden_size> <n_epochs> <learning_rate> <dropout>
   ```

2. **Run `train_lstm.sh`**:
   ```bash
   cd SAR_EMERGENCE_RESEARCH/lstm/scripts
   sbatch train_lstm.sh
   This will run train_w_stats.py on the NJIT Wulver HPC.
   ```

## Evaluation Pipeline

### Transformer Evaluation: configured on wandb

### LSTM Evaluation
```bash
cd SAR_EMERGENCE_RESEARCH/lstm
python eval.py
```

## Data Structure

The project uses SAR data organized by Active Region (AR) number:
data/
├── AR11698/
├── AR11726/
├── AR13165/
├── AR13179/
└── AR13183/
...


Each AR directory contains:
- Power maps (`mean_pmdop{AR}_flat.npz`)
- Magnetic flux data (`mean_mag{AR}_flat.npz`)
- Intensity data (`mean_int{AR}_flat.npz`)

## Model Outputs

### Transformer Models
Saved in: `transformer/results/batch_size_search/`
Format: `transformer_t{time_window}_r{rid_of_top}_i{num_in}_n{num_layers}_h{hidden_size}_e{epochs}_bs{batch_size}_l{learning_rate}_d{dropout}.pth`

### LSTM Models
Saved in: `lstm/results/`
Format: `t{time_window}_r{rid_of_top}_i{num_in}_n{num_layers}_h{hidden_size}_e{epochs}_l{learning_rate}_d{dropout}.pth`

## Weights & Biases Integration

The project uses Weights & Biases for experiment tracking. Configure your W&B credentials in the `.env` file as shown in the [Environment Configuration](#wulver-cluster-setup) section.

Training metrics, model artifacts, and experiment results are automatically logged to your W&B project dashboard.