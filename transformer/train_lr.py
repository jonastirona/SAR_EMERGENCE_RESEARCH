import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from typing import Dict, List, Tuple, Any
import logging
import sys
from datetime import datetime
import os
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import math

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Import from existing files
from transformer.models.st_transformer import SpatioTemporalTransformer
from transformer.functions import lstm_ready, smooth_with_numpy, emergence_indication, split_sequences

# Import the new evaluation module
from transformer.eval import evaluate_models_for_ar

# Set up logging
logging.basicConfig(
    level=logging.WARNING,  # Changed from INFO to WARNING
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class ConstantScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Simple constant learning rate scheduler - no warmup, no decay"""
    def __init__(self, optimizer, last_epoch=-1):
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # Always return the base learning rates unchanged
        return self.base_lrs

def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate R² score."""
    if len(y_true.shape) > 1:
        y_true = y_true.flatten()
    if len(y_pred.shape) > 1:
        y_pred = y_pred.flatten()
    return r2_score(y_true, y_pred)

def calculate_derivative(time_series: np.ndarray, time_step: float = 1.0) -> np.ndarray:
    """Calculate the derivative of a time series."""
    return np.gradient(time_series, time_step)

def find_negative_derivative_periods(derivative: np.ndarray, min_duration: int = 4) -> List[Tuple[int, int]]:
    """Find periods where derivative remains negative for at least min_duration hours."""
    periods = []
    start_idx = None
    current_negative_duration = 0
    
    # Convert derivative to boolean array for negative values
    is_negative = derivative < 0
    
    for i in range(len(derivative)):
        if is_negative[i]:
            if start_idx is None:
                start_idx = i
            current_negative_duration += 1
        else:
            if current_negative_duration >= min_duration:
                periods.append((start_idx, i))
            start_idx = None
            current_negative_duration = 0
    
    # Check if we end with a negative period
    if start_idx is not None and current_negative_duration >= min_duration:
        periods.append((start_idx, len(derivative)))
    
    return periods

def detect_emergence_window(observed: np.ndarray, predicted: np.ndarray, 
                          time_step: float = 1.0,
                          negative_duration: int = 4,
                          window_size: int = 24) -> Tuple[int, int]:
    """Detect the emergence window as a 24-hour period centered around the first detected emergence criteria."""
    # Calculate derivative of observed series
    d_obs = calculate_derivative(observed, time_step)
    
    # Find the FIRST point where emergence is detected (first point of 4-hour sustained negative derivative)
    first_emergence_point = None
    current_negative_duration = 0
    potential_start = None
    
    # Convert derivative to boolean array for negative values
    is_negative = d_obs < -0.01  # Use the same threshold as emergence_indication
    
    for i in range(len(d_obs)):
        if is_negative[i]:
            if potential_start is None:
                potential_start = i
            current_negative_duration += 1
            
            # If we've sustained 4 hours of negative derivative, mark the FIRST point
            if current_negative_duration >= negative_duration:
                first_emergence_point = potential_start  # Use the start of the 4-hour period
                break
        else:
            potential_start = None
            current_negative_duration = 0
    
    # If no emergence detected, use the first 24 hours
    if first_emergence_point is None:
        return 0, min(window_size, len(observed))
    
    # Create ±12 hour window around the first emergence point (24 hours total)
    half_window = window_size // 2  # 12 hours
    window_start = first_emergence_point - half_window
    window_end = first_emergence_point + half_window
    
    # Ensure we stay within bounds and maintain 24-hour window
    if window_start < 0:
        # If we go before start, shift window forward but keep 24 hours
        window_start = 0
        window_end = min(len(observed), window_size)
    elif window_end > len(observed):
        # If we go past end, shift window backward but keep 24 hours
        window_end = len(observed)
        window_start = max(0, len(observed) - window_size)
    
    return window_start, window_end

def calculate_emergence_metrics(observed: np.ndarray, predicted: np.ndarray, 
                              time_step: float = 1.0) -> Dict[str, float]:
    """Calculate metrics specific to emergence detection."""
    # Convert tuples to numpy arrays if needed
    if isinstance(observed, tuple):
        observed = np.array(observed)
    if isinstance(predicted, tuple):
        predicted = np.array(predicted)
    
    # Ensure we have 1D arrays for time series analysis
    if len(observed.shape) > 1:
        observed = observed.flatten()
    if len(predicted.shape) > 1:
        predicted = predicted.flatten()
    
    # Enforce matching lengths by truncating to the smallest length
    min_len = min(len(observed), len(predicted))
    observed = observed[:min_len]
    predicted = predicted[:min_len]
    
    # Check for minimum size requirement
    if min_len < 24:
        print(f"Warning: Arrays too small for meaningful analysis (size: {min_len})")
        # Return default metrics
        return {
            'emergence_rmse': float('nan'),
            'emergence_mae': float('nan'),
            'emergence_mse': float('nan'),
            'emergence_r2': float('nan'),
            'emergence_window_start': 0,
            'emergence_window_end': min_len,
            'emergence_window_size': min_len,
            'avg_negative_derivative_duration': 0.0,
            'num_negative_derivative_periods': 0,
            'obs_derivative_avg': 0.0,
            'pred_derivative_avg': 0.0,
            'emergence_time_diff': 0.0,
            'overall_rmse': float('nan'),
            'overall_mae': float('nan'),
            'overall_mse': float('nan'),
            'overall_r2': float('nan')
        }
    
    # Calculate derivatives
    d_obs = calculate_derivative(observed, time_step)
    d_pred = calculate_derivative(predicted, time_step)
    
    # Find negative derivative periods
    obs_periods = find_negative_derivative_periods(d_obs)
    pred_periods = find_negative_derivative_periods(d_pred)
    
    # Calculate time differences between corresponding periods
    time_diffs = []
    for obs_start, obs_end in obs_periods:
        for pred_start, pred_end in pred_periods:
            # Calculate overlap
            overlap_start = max(obs_start, pred_start)
            overlap_end = min(obs_end, pred_end)
            if overlap_end > overlap_start:
                time_diffs.append(overlap_end - overlap_start)
    
    # Detect emergence window
    start_idx, end_idx = detect_emergence_window(observed, predicted, time_step)
    
    # Calculate metrics for the detected emergence window
    emergence_window_observed = observed[start_idx:end_idx]
    emergence_window_predicted = predicted[start_idx:end_idx]
    
    emergence_rmse = np.sqrt(np.mean((emergence_window_observed - emergence_window_predicted)**2))
    emergence_mae = np.mean(np.abs(emergence_window_observed - emergence_window_predicted))
    emergence_mse = np.mean((emergence_window_observed - emergence_window_predicted)**2)
    emergence_r2 = calculate_r2(emergence_window_observed, emergence_window_predicted)
    
    # Calculate additional metrics
    window_size = end_idx - start_idx
    obs_derivative_avg = np.mean(d_obs[start_idx:end_idx])
    pred_derivative_avg = np.mean(d_pred[start_idx:end_idx])
    
    # Calculate the time difference between observed and predicted emergence
    obs_emergence_start = start_idx
    pred_emergence_start = None
    
    # Find the first 4-hour negative derivative period in predicted series
    current_negative_duration = 0
    is_negative = d_pred < 0
    
    for i in range(len(d_pred)):
        if is_negative[i]:
            if pred_emergence_start is None:
                pred_emergence_start = i
            current_negative_duration += 1
        else:
            if current_negative_duration >= 4:
                break
            pred_emergence_start = None
            current_negative_duration = 0
    
    emergence_time_diff = (pred_emergence_start - obs_emergence_start) if pred_emergence_start is not None else None
    
    # Calculate overall metrics
    overall_rmse = np.sqrt(np.mean((observed - predicted)**2))
    overall_mae = np.mean(np.abs(observed - predicted))
    overall_mse = np.mean((observed - predicted)**2)
    overall_r2 = calculate_r2(observed, predicted)
    
    return {
        'emergence_rmse': float(emergence_rmse),
        'emergence_mae': float(emergence_mae),
        'emergence_mse': float(emergence_mse),
        'emergence_r2': float(emergence_r2),
        'emergence_window_start': int(start_idx),
        'emergence_window_end': int(end_idx),
        'emergence_window_size': int(window_size),
        'avg_negative_derivative_duration': float(np.mean(time_diffs)) if time_diffs else 0.0,
        'num_negative_derivative_periods': int(len(time_diffs)),
        'obs_derivative_avg': float(obs_derivative_avg),
        'pred_derivative_avg': float(pred_derivative_avg),
        'emergence_time_diff': float(emergence_time_diff) if emergence_time_diff is not None else 0.0,
        'overall_rmse': float(overall_rmse),
        'overall_mae': float(overall_mae),
        'overall_mse': float(overall_mse),
        'overall_r2': float(overall_r2)
    }

def calculate_tile_level_emergence_metrics(observed: np.ndarray, predicted: np.ndarray, 
                                         tile_indices: np.ndarray, 
                                         time_step: float = 1.0) -> Dict[str, float]:
    """
    Calculate emergence metrics per tile and aggregate them.
    
    Args:
        observed: Array of shape (n_samples, sequence_length) with observed values
        predicted: Array of shape (n_samples, sequence_length) with predicted values  
        tile_indices: Array of shape (n_samples,) indicating which tile each sample belongs to
        time_step: Time step for derivative calculation
        
    Returns:
        Dictionary of aggregated metrics across all tiles
    """
    # Get unique tiles
    unique_tiles = np.unique(tile_indices)
    
    # Initialize lists to store per-tile metrics
    tile_metrics = {
        'emergence_rmse': [],
        'emergence_mae': [],
        'emergence_mse': [],
        'emergence_r2': [],
        'emergence_time_diff': [],
        'overall_rmse': [],
        'overall_mae': [], 
        'overall_mse': [],
        'overall_r2': [],
        'emergence_window_size': [],
        'num_valid_tiles': 0
    }
    
    valid_tiles = 0
    
    for tile_idx in unique_tiles:
        # Get all samples for this tile
        tile_mask = tile_indices == tile_idx
        tile_observed = observed[tile_mask]
        tile_predicted = predicted[tile_mask]
        
        # Skip if not enough samples for this tile
        if len(tile_observed) == 0:
            continue
            
        # Flatten sequences for time series analysis
        # Each sample is a sequence, concatenate them for tile-level analysis
        tile_obs_flat = tile_observed.flatten()
        tile_pred_flat = tile_predicted.flatten()
        
        # Skip tiles with insufficient data
        if len(tile_obs_flat) < 24:
            continue
            
        try:
            # Calculate emergence metrics for this tile
            metrics = calculate_emergence_metrics(tile_obs_flat, tile_pred_flat, time_step)
            
            # Only include tiles where emergence metrics could be calculated
            if not np.isnan(metrics['emergence_rmse']):
                tile_metrics['emergence_rmse'].append(metrics['emergence_rmse'])
                tile_metrics['emergence_mae'].append(metrics['emergence_mae'])
                tile_metrics['emergence_mse'].append(metrics['emergence_mse'])
                tile_metrics['emergence_r2'].append(metrics['emergence_r2'])
                tile_metrics['emergence_time_diff'].append(metrics['emergence_time_diff'])
                tile_metrics['overall_rmse'].append(metrics['overall_rmse'])
                tile_metrics['overall_mae'].append(metrics['overall_mae'])
                tile_metrics['overall_mse'].append(metrics['overall_mse'])
                tile_metrics['overall_r2'].append(metrics['overall_r2'])
                tile_metrics['emergence_window_size'].append(metrics['emergence_window_size'])
                valid_tiles += 1
                
        except Exception as e:
            continue
    
    tile_metrics['num_valid_tiles'] = valid_tiles
    
    # Aggregate metrics across tiles
    aggregated_metrics = {}
    
    for metric_name in ['emergence_rmse', 'emergence_mae', 'emergence_mse', 'emergence_r2',
                       'emergence_time_diff', 'overall_rmse', 'overall_mae', 'overall_mse', 
                       'overall_r2', 'emergence_window_size']:
        
        values = tile_metrics[metric_name]
        if len(values) > 0:
            # Filter out any remaining NaN values
            valid_values = [v for v in values if not np.isnan(v)]
            if len(valid_values) > 0:
                aggregated_metrics[metric_name] = float(np.mean(valid_values))
            else:
                aggregated_metrics[metric_name] = float('nan')
        else:
            aggregated_metrics[metric_name] = float('nan')
    
    # Add summary statistics
    aggregated_metrics['num_valid_tiles'] = valid_tiles
    aggregated_metrics['total_tiles'] = len(unique_tiles)
    aggregated_metrics['tile_success_rate'] = valid_tiles / len(unique_tiles) if len(unique_tiles) > 0 else 0.0
    
    # Also return individual timing differences for box plot analysis
    individual_timing_diffs = tile_metrics['emergence_time_diff']
    aggregated_metrics['individual_timing_diffs'] = individual_timing_diffs
    
    return aggregated_metrics

def load_all_ars_data(ARs, rid_of_top, size, num_in, num_pred):
    all_inputs = []
    all_intensities = []
    
    # Collect all raw data first to compute global normalization stats
    all_stacked_maps = []
    all_mag_flux = []
    all_intensities_raw = []
    
    for AR in ARs:
        power_maps = np.load(f'/mmfs1/project/mx6/jst26/SAR_EMERGENCE_RESEARCH/data/AR{AR}/mean_pmdop{AR}_flat.npz', allow_pickle=True)
        mag_flux = np.load(f'/mmfs1/project/mx6/jst26/SAR_EMERGENCE_RESEARCH/data/AR{AR}/mean_mag{AR}_flat.npz', allow_pickle=True)
        intensities = np.load(f'/mmfs1/project/mx6/jst26/SAR_EMERGENCE_RESEARCH/data/AR{AR}/mean_int{AR}_flat.npz', allow_pickle=True)
        power_maps23 = power_maps['arr_0']
        power_maps34 = power_maps['arr_1']
        power_maps45 = power_maps['arr_2']
        power_maps56 = power_maps['arr_3']
        mag_flux = mag_flux['arr_0']
        intensities = intensities['arr_0']
        # Trim array to get rid of top and bottom 0 tiles
        power_maps23 = power_maps23[rid_of_top*size:-rid_of_top*size, :]
        power_maps34 = power_maps34[rid_of_top*size:-rid_of_top*size, :]
        power_maps45 = power_maps45[rid_of_top*size:-rid_of_top*size, :]
        power_maps56 = power_maps56[rid_of_top*size:-rid_of_top*size, :]
        mag_flux = mag_flux[rid_of_top*size:-rid_of_top*size, :]; mag_flux[np.isnan(mag_flux)] = 0
        intensities = intensities[rid_of_top*size:-rid_of_top*size, :]; intensities[np.isnan(intensities)] = 0
        # stack inputs
        stacked_maps = np.stack([power_maps23, power_maps34, power_maps45, power_maps56], axis=1); stacked_maps[np.isnan(stacked_maps)] = 0
        
        all_stacked_maps.append(stacked_maps)
        all_mag_flux.append(mag_flux)
        all_intensities_raw.append(intensities)
    
    # Compute global normalization statistics
    all_stacked_concat = np.concatenate(all_stacked_maps, axis=0)
    all_mag_concat = np.concatenate(all_mag_flux, axis=0)
    all_int_concat = np.concatenate(all_intensities_raw, axis=0)
    
    global_min_p = np.min(all_stacked_concat)
    global_max_p = np.max(all_stacked_concat)
    global_min_m = np.min(all_mag_concat)
    global_max_m = np.max(all_mag_concat)
    global_min_i = np.min(all_int_concat)
    global_max_i = np.max(all_int_concat)
    
    # Store global stats for later use in evaluation
    global GLOBAL_NORM_STATS
    GLOBAL_NORM_STATS = {
        'min_p': global_min_p, 'max_p': global_max_p,
        'min_m': global_min_m, 'max_m': global_max_m,
        'min_i': global_min_i, 'max_i': global_max_i
    }
    
    print(f"Global normalization stats computed:")
    print(f"  Power maps: [{global_min_p:.4f}, {global_max_p:.4f}]")
    print(f"  Mag flux: [{global_min_m:.4f}, {global_max_m:.4f}]")
    print(f"  Intensities: [{global_min_i:.4f}, {global_max_i:.4f}]")
    
    # Now normalize each AR using global statistics
    for i, AR in enumerate(ARs):
        stacked_maps = all_stacked_maps[i]
        mag_flux = all_mag_flux[i]
        intensities = all_intensities_raw[i]
        
        # Apply global normalization
        stacked_maps = (stacked_maps - global_min_p) / (global_max_p - global_min_p)
        mag_flux = (mag_flux - global_min_m) / (global_max_m - global_min_m)
        intensities = (intensities - global_min_i) / (global_max_i - global_min_i)
        
        # Reshape mag_flux to have an extra dimension and then put it with pmaps
        mag_flux_reshaped = np.expand_dims(mag_flux, axis=1)
        pm_and_flux = np.concatenate([stacked_maps, mag_flux_reshaped], axis=1)
        # append all ARs
        all_inputs.append(pm_and_flux)
        all_intensities.append(intensities)
    
    all_inputs = np.stack(all_inputs, axis=-1)
    all_intensities = np.stack(all_intensities, axis=-1)
    
    return all_inputs, all_intensities

# Global variable to store normalization statistics
GLOBAL_NORM_STATS = None

def run_single_experiment(config: Dict[str, Any], device: torch.device, learning_rate: float, global_step_offset: int = 0) -> Dict[str, float]:
    """Run a single experiment with the given configuration."""
    print(f"Training model with learning rate {learning_rate}...")
    
    # Initialize model with fixed attention heads
    model = SpatioTemporalTransformer(
        input_dim=5,  # 4 power maps + 1 magnetic flux
        seq_len=config['num_in'],
        embed_dim=config['embed_dim'],
        num_heads=4,  # Fixed at 4 heads
        ff_dim=config['ff_dim'],
        num_layers=config['num_layers'],  # Fixed at 3 layers
        output_dim=config['num_pred'],
        dropout=config['dropout'],
    ).to(device)
    
    print(f"\n=== MODEL ARCHITECTURE ===")
    print(f"Model: SpatioTemporalTransformer")
    print(f"Input dim: 5 (4 power maps + 1 magnetic flux)")
    print(f"Sequence length: {config['num_in']}")
    print(f"Embed dim: {config['embed_dim']}")
    print(f"Attention heads: 4 (fixed)")
    print(f"Feed-forward dim: {config['ff_dim']}")
    print(f"Number of layers: {config['num_layers']} (fixed)")
    print(f"Output dim: {config['num_pred']}")
    print(f"Dropout: {config['dropout']}")
    print(f"Learning rate: {learning_rate}")
    print(f"Device: {device}")
    
    # Count model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB (assuming 4 bytes per param)")
    
    # ARs list (copied from train_w_stats.py)
    ARs = [11130,11149,11158,11162,11199,11327,11344,11387,11393,11416,11422,11455,11619,11640,11660,11678,11682,11765,11768,11776,11916,11928,12036,12051,12085,12089,12144,12175,12203,12257,12331,12494,12659,12778,12864,12877,12900,12929,13004,13085,13098,13179]
    size = 9
    rid_of_top = config['rid_of_top']
    num_in = config['num_in']
    num_pred = config['num_pred']

    all_inputs, all_intensities = load_all_ars_data(ARs, rid_of_top, size, num_in, num_pred)
    input_size = np.shape(all_inputs)[1]
    
    print(f"\n=== DATA SHAPES ===")
    print(f"all_inputs shape: {all_inputs.shape}")  # (tiles, features, time, ARs)
    print(f"all_intensities shape: {all_intensities.shape}")  # (tiles, time, ARs)
    print(f"input_size (features): {input_size}")

    # Prepare data for all tiles and ARs - keep track of tile indices
    X_trains = []
    y_trains = []
    tile_indices = []  # Track which tile each sample comes from
    remaining_rows = size - 2*rid_of_top
    tiles = remaining_rows * size
    
    print(f"\n=== PROCESSING INFO ===")
    print(f"Number of ARs: {len(ARs)}")
    print(f"Grid size: {size}x{size}")
    print(f"Tiles after trimming: {tiles} (removed {rid_of_top} from top/bottom)")
    print(f"Input sequence length: {num_in}")
    print(f"Prediction sequence length: {num_pred}")
    
    for ar_idx in range(len(ARs)):
        power_maps = all_inputs[:,:,:,ar_idx]
        intensities = all_intensities[:,:,ar_idx]
        print(f"\nAR {ARs[ar_idx]} - power_maps shape: {power_maps.shape}, intensities shape: {intensities.shape}")
        
        for tile in range(tiles):
            X_tile, y_tile = lstm_ready(tile, size, power_maps, intensities, num_in, num_pred, model_seq_len=num_in)
            X_trains.append(X_tile)
            y_trains.append(y_tile)
            # Add tile indices for each sequence in this tile
            tile_indices.extend([tile] * len(X_tile))
            
            # Print shape info for first few tiles
            if tile < 3 and ar_idx == 0:
                print(f"  Tile {tile}: X_tile shape: {X_tile.shape}, y_tile shape: {y_tile.shape}")
    
    X = torch.cat(X_trains, dim=0)
    y = torch.cat(y_trains, dim=0)
    X = torch.reshape(X, (X.shape[0], num_in, X.shape[2]))
    tile_indices = np.array(tile_indices)
    
    print(f"\n=== FINAL TENSOR SHAPES ===")
    print(f"X (input) shape: {X.shape}")  # (total_samples, seq_len, features)
    print(f"y (target) shape: {y.shape}")  # (total_samples, num_pred)
    print(f"tile_indices shape: {tile_indices.shape}")  # (total_samples,)
    print(f"Total samples: {len(X)}")

    # Split data first on CPU
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    tile_indices_train, tile_indices_test = tile_indices[:train_size], tile_indices[train_size:]
    
    print(f"\n=== TRAIN/TEST SPLIT ===")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Create DataLoaders with CPU data
    batch_size = min(64, len(X_train) // 10)  # Adaptive batch size
    print(f"Batch size: {batch_size}")
    
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    
    # Move data to device after DataLoader creation
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ConstantScheduler(optimizer)
    loss_fn = nn.MSELoss()
    
    # Initialize tracking variables
    best_test_loss = float('inf')
    best_predictions = None
    best_observations = None
    best_model_state = None
    best_tile_indices = None
    
    # Initialize lists to store per-epoch metrics
    train_losses = []
    test_losses = []
    test_emergence_rmses = []  # Only track test emergence RMSE for plotting
    test_emergence_mses = []   # Track emergence MSE across epochs
    test_overall_mses = []     # Track overall MSE across epochs
    
    for epoch in range(config['n_epochs']):
        # Training phase
        model.train()
        epoch_train_loss = 0
        num_batches = 0
        
        for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
            # Move batch to device
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Print shapes for first batch of first epoch
            if epoch == 0 and batch_idx == 0:
                print(f"\n=== FIRST BATCH SHAPES ===")
                print(f"Input batch shape: {batch_X.shape}")  # (batch_size, seq_len, features)
                print(f"Target batch shape: {batch_y.shape}")  # (batch_size, output_dim)
                print(f"Input range: [{batch_X.min().item():.4f}, {batch_X.max().item():.4f}]")
                print(f"Target range: [{batch_y.min().item():.4f}, {batch_y.max().item():.4f}]")
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            
            # Print output shapes for first batch of first epoch
            if epoch == 0 and batch_idx == 0:
                print(f"Model output shape: {outputs.shape}")  # (batch_size, output_dim)
                print(f"Output range: [{outputs.min().item():.4f}, {outputs.max().item():.4f}]")
                print(f"Expected vs actual output shapes: {batch_y.shape} vs {outputs.shape}")
                print(f"Shapes match: {batch_y.shape == outputs.shape}")
            
            loss = loss_fn(outputs, batch_y)
            
            # Check for NaN loss
            if torch.isnan(loss):
                print(f"Warning: NaN loss detected at epoch {epoch}, batch {batch_idx}")
                continue
                
            loss.backward()
            
            # Add gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_train_loss += loss.item()
            num_batches += 1
        
        # Calculate average training loss
        if num_batches > 0:
            epoch_train_loss /= num_batches
        else:
            epoch_train_loss = float('nan')
        
        # Evaluation phase
        model.eval()
        epoch_test_loss = 0
        num_test_batches = 0
        all_test_preds = []
        all_test_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                # Move batch to device
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                test_pred = model(batch_X)
                test_loss = loss_fn(test_pred, batch_y)
                
                if not torch.isnan(test_loss):
                    epoch_test_loss += test_loss.item()
                    num_test_batches += 1
                    all_test_preds.append(test_pred.cpu())
                    all_test_targets.append(batch_y.cpu())
            
            # Calculate average test loss
            if num_test_batches > 0:
                epoch_test_loss /= num_test_batches
                
                # Concatenate all predictions and targets for metrics calculation
                all_test_preds = torch.cat(all_test_preds, dim=0).numpy()
                all_test_targets = torch.cat(all_test_targets, dim=0).numpy()
                
                # Calculate tile-level emergence metrics for test set
                test_emergence_metrics = calculate_tile_level_emergence_metrics(
                    all_test_targets, 
                    all_test_preds,
                    tile_indices_test
                )
            else:
                epoch_test_loss = float('nan')
                test_emergence_metrics = {'emergence_rmse': float('nan')}
                all_test_preds = y_test.cpu().numpy()
                all_test_targets = y_test.cpu().numpy()
            
            # Store metrics for plotting
            train_losses.append(epoch_train_loss)
            test_losses.append(epoch_test_loss)
            test_emergence_rmses.append(test_emergence_metrics['emergence_rmse'])
            test_emergence_mses.append(test_emergence_metrics['emergence_mse'])
            test_overall_mses.append(test_emergence_metrics['overall_mse'])
            
            # Calculate unique global step for this trial and epoch
            global_step = global_step_offset + epoch
            
            # Log per-epoch metrics with lr prefix and unique global step
            lr_str = f"{learning_rate:.4f}"
            metrics_to_log = {
                f'lr_{lr_str}/train_loss': epoch_train_loss,
                f'lr_{lr_str}/test_loss': epoch_test_loss,
                f'lr_{lr_str}/test_emergence_rmse': test_emergence_metrics['emergence_rmse'],
                f'lr_{lr_str}/test_emergence_mse': test_emergence_metrics.get('emergence_mse', float('nan')),
                f'lr_{lr_str}/test_emergence_mae': test_emergence_metrics.get('emergence_mae', float('nan')),
                f'lr_{lr_str}/test_emergence_r2': test_emergence_metrics.get('emergence_r2', float('nan')),
                f'lr_{lr_str}/test_overall_rmse': test_emergence_metrics.get('overall_rmse', float('nan')),
                f'lr_{lr_str}/test_overall_mse': test_emergence_metrics.get('overall_mse', float('nan')),
                f'lr_{lr_str}/test_overall_mae': test_emergence_metrics.get('overall_mae', float('nan')),
                f'lr_{lr_str}/test_overall_r2': test_emergence_metrics.get('overall_r2', float('nan')),
                f'lr_{lr_str}/learning_rate': optimizer.param_groups[0]['lr'],
                f'lr_{lr_str}/epoch': epoch,
                # Also log without lr prefix for easy cross-trial comparison
                'train_loss': epoch_train_loss,
                'test_loss': epoch_test_loss,
                'test_emergence_rmse': test_emergence_metrics['emergence_rmse'],
                'test_emergence_mse': test_emergence_metrics.get('emergence_mse', float('nan')),
                'test_emergence_mae': test_emergence_metrics.get('emergence_mae', float('nan')),
                'test_emergence_r2': test_emergence_metrics.get('emergence_r2', float('nan')),
                'test_overall_rmse': test_emergence_metrics.get('overall_rmse', float('nan')),
                'test_overall_mse': test_emergence_metrics.get('overall_mse', float('nan')),
                'test_overall_mae': test_emergence_metrics.get('overall_mae', float('nan')),
                'test_overall_r2': test_emergence_metrics.get('overall_r2', float('nan')),
                'learning_rate': optimizer.param_groups[0]['lr'],
                'epoch': epoch,
                'current_lr': learning_rate,  # Add current LR for filtering
                'trial_id': f'lr_{lr_str}'  # Add trial identifier
            }
            
            # Log with unique global step to avoid conflicts
            wandb.log(metrics_to_log, step=global_step)
            
            if not np.isnan(epoch_test_loss) and epoch_test_loss < best_test_loss:
                best_test_loss = epoch_test_loss
                best_predictions = all_test_preds
                best_observations = all_test_targets
                best_model_state = model.state_dict()
                best_tile_indices = tile_indices_test
        
        scheduler.step()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch + 1}/{config['n_epochs']} - Train Loss: {epoch_train_loss:.5f}, Test Loss: {epoch_test_loss:.5f}, LR: {current_lr:.2e}")
    
    # Calculate final tile-level emergence metrics on best model
    final_metrics = calculate_tile_level_emergence_metrics(
        best_observations, 
        best_predictions,
        best_tile_indices
    )
    
    # Add lr info to metrics
    final_metrics['learning_rate'] = learning_rate
    
    # Return all metrics including the lists for plotting and best model state
    return {
        **final_metrics,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_emergence_losses': test_emergence_rmses,  # Use test for consistency
        'test_emergence_losses': test_emergence_rmses,
        'test_emergence_mses': test_emergence_mses,      # Add emergence MSE across epochs
        'test_overall_mses': test_overall_mses,          # Add overall MSE across epochs
        'best_model_state': best_model_state
    }

def create_cross_trial_comparison_plots(all_results: Dict[float, Dict], config: Dict) -> None:
    """Create comprehensive comparison plots across all learning rate trials with separate grids for each metric."""
    
    # Extract data for plotting
    learning_rates = sorted(all_results.keys())
    
    # Create two comprehensive figures - one for overall metrics, one for emergence metrics
    
    # 1. Overall Performance Metrics - 2x2 grid
    fig1, axes1 = plt.subplots(2, 2, figsize=(20, 14))  # Increased figure size
    fig1.suptitle('Overall Performance Metrics Across Learning Rates (3 Layers, 4 Heads)', fontsize=18, y=0.95)
    
    overall_metrics = ['overall_mse', 'overall_rmse', 'overall_mae', 'overall_r2']
    metric_labels = ['MSE', 'RMSE', 'MAE', 'R²']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Flatten axes for easier iteration
    axes1_flat = axes1.flatten()
    
    # Calculate appropriate bar width based on number of learning rates
    bar_width = min(0.6, 1.2 / len(learning_rates))  # Adaptive bar width - increased for thicker bars
    
    for i, (metric, label, color) in enumerate(zip(overall_metrics, metric_labels, colors)):
        ax = axes1_flat[i]
        values = [all_results[lr][metric] for lr in learning_rates]
        
        # Create bar plot with adaptive width
        bars = ax.bar(range(len(learning_rates)), values, color=color, alpha=0.8, width=bar_width)
        
        # Add value labels on bars with minimal offset - directly on top
        max_val = max(values)
        min_val = min(values)
        val_range = max_val - min_val
        text_offset = max_val * 0.02  # Much smaller offset - just 2% of max value
        
        for j, (bar, value) in enumerate(zip(bars, values)):
            height = bar.get_height()
            # Position text just above bars with minimal offset
            ax.text(j, height + text_offset,
                    f'{value:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Learning Rate', fontsize=12)
        ax.set_ylabel(f'{label} Value', fontsize=12)
        ax.set_title(f'Overall {label}', fontsize=14, pad=20)
        
        # Set x-axis ticks and labels
        ax.set_xticks(range(len(learning_rates)))
        ax.set_xticklabels([f'{lr:.4f}' for lr in learning_rates])
        
        # Adjust y-axis limits to accommodate text with less space
        ax.set_ylim([min_val - val_range * 0.1, max_val + val_range * 0.15])  # Reduced top margin
        ax.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(range(len(learning_rates)), values, 1)
        p = np.poly1d(z)
        ax.plot(range(len(learning_rates)), p(range(len(learning_rates))), "--", alpha=0.7, color='red', linewidth=2)
        
        # Add trend line formula to corner
        slope, intercept = z[0], z[1]
        formula = f'y = {slope:.4f}x + {intercept:.4f}'
        ax.text(0.02, 0.98, formula, transform=ax.transAxes, fontsize=8, 
                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.93])  # Better spacing
    wandb.log({"overall_metrics_comparison": wandb.Image(fig1)})
    plt.close(fig1)
    
    # 2. Emergence Window Performance Metrics - 2x2 grid  
    fig2, axes2 = plt.subplots(2, 2, figsize=(20, 14))  # Increased figure size
    fig2.suptitle('Emergence Window Performance Metrics (±12 hrs) Across Learning Rates (3 Layers, 4 Heads)', fontsize=18, y=0.95)
    
    emergence_metrics = ['emergence_mse', 'emergence_rmse', 'emergence_mae', 'emergence_r2']
    
    # Flatten axes for easier iteration
    axes2_flat = axes2.flatten()
    
    for i, (metric, label, color) in enumerate(zip(emergence_metrics, metric_labels, colors)):
        ax = axes2_flat[i]
        values = [all_results[lr][metric] for lr in learning_rates]
        
        # Create bar plot with adaptive width
        bars = ax.bar(range(len(learning_rates)), values, color=color, alpha=0.8, width=bar_width)
        
        # Add value labels on bars with minimal offset - directly on top
        max_val = max(values)
        min_val = min(values)
        val_range = max_val - min_val
        text_offset = max_val * 0.02  # Much smaller offset - just 2% of max value
        
        for j, (bar, value) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(j, height + text_offset,
                    f'{value:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Learning Rate', fontsize=12)
        ax.set_ylabel(f'{label} Value', fontsize=12)
        ax.set_title(f'Emergence Window {label}', fontsize=14, pad=20)
        
        # Set x-axis ticks and labels
        ax.set_xticks(range(len(learning_rates)))
        ax.set_xticklabels([f'{lr:.4f}' for lr in learning_rates])
        
        # Adjust y-axis limits to accommodate text with less space
        ax.set_ylim([min_val - val_range * 0.1, max_val + val_range * 0.15])  # Reduced top margin
        ax.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(range(len(learning_rates)), values, 1)
        p = np.poly1d(z)
        ax.plot(range(len(learning_rates)), p(range(len(learning_rates))), "--", alpha=0.7, color='red', linewidth=2)
        
        # Add trend line formula to corner
        slope, intercept = z[0], z[1]
        formula = f'y = {slope:.4f}x + {intercept:.4f}'
        ax.text(0.02, 0.98, formula, transform=ax.transAxes, fontsize=8, 
                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.93])  # Better spacing
    wandb.log({"emergence_metrics_comparison": wandb.Image(fig2)})
    plt.close(fig2)
    
    # 3. Emergence Timing Accuracy - strip plots showing distributions
    fig3, ax3 = plt.subplots(1, 1, figsize=(14, 8))
    fig3.suptitle('Emergence Timing Difference Distributions Across Learning Rates (3 Layers, 4 Heads)', fontsize=16, y=0.95)
    
    # Collect individual timing differences for each learning rate
    timing_data = []
    timing_stats = []
    
    print("DEBUG: Examining timing difference data for strip plots...")
    
    for lr in learning_rates:
        individual_diffs = all_results[lr].get('individual_timing_diffs', [])
        # Filter out None and NaN values
        valid_diffs = [x for x in individual_diffs if x is not None and not np.isnan(x)]
        timing_data.append(valid_diffs)
        
        print(f"  LR {lr}: {len(individual_diffs)} total, {len(valid_diffs)} valid timing differences")
        if valid_diffs:
            print(f"    Range: [{min(valid_diffs):.1f}, {max(valid_diffs):.1f}] hours")
            print(f"    Sample values: {valid_diffs[:5]}")
        
        if valid_diffs:
            mean_val = np.mean(valid_diffs)
            std_val = np.std(valid_diffs)
            timing_stats.append({'mean': mean_val, 'std': std_val, 'count': len(valid_diffs)})
        else:
            timing_stats.append({'mean': np.nan, 'std': np.nan, 'count': 0})
    
    # Create strip plots with mean and std
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    
    # Check if we have any valid data
    has_any_data = any(len(data) > 0 for data in timing_data)
    
    if has_any_data:
        # Plot individual data points with jitter
        for i, (lr, data, stats) in enumerate(zip(learning_rates, timing_data, timing_stats)):
            if len(data) > 0:
                # Add jitter to x-coordinates for visibility
                jitter_amount = 0.15
                jitter = np.random.uniform(-jitter_amount, jitter_amount, len(data))
                x_positions = np.full(len(data), i) + jitter
                
                # Plot individual points
                color = colors[i % len(colors)]
                ax3.scatter(x_positions, data, alpha=0.6, s=40, color=color, 
                           edgecolors='white', linewidth=0.5, zorder=3, label=f'LR {lr:.4f}')
                
                # Plot mean as a larger marker
                if not np.isnan(stats['mean']):
                    ax3.scatter(i, stats['mean'], s=120, color='black', marker='D', 
                               edgecolors='white', linewidth=2, zorder=5)
                    ax3.scatter(i, stats['mean'], s=80, color=color, marker='D', 
                               edgecolors='black', linewidth=1, zorder=6)
                    
                    # Plot standard deviation as error bars
                    if stats['count'] > 1 and not np.isnan(stats['std']):
                        ax3.errorbar(i, stats['mean'], yerr=stats['std'], 
                                   color='black', capsize=8, capthick=2, linewidth=2, 
                                   alpha=0.8, zorder=4)
        
        # Add mean and std annotations
        for i, (lr, stats) in enumerate(zip(learning_rates, timing_stats)):
            if not np.isnan(stats['mean']):
                # Position text above the highest points
                max_val = max([max(data) if data else stats['mean'] for data in timing_data])
                y_pos = max_val + abs(max_val) * 0.15 if max_val != 0 else stats['mean'] + 3
                
                text = f"μ={stats['mean']:.1f}±{stats['std']:.1f}h\n(n={stats['count']})"
                ax3.text(i, y_pos, text, ha='center', va='bottom', 
                        fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                                alpha=0.9, edgecolor='gray', linewidth=1))
        
        plot_type = "Strip Plot"
        
    else:
        # No valid data - show message
        ax3.text(0.5, 0.5, 'No Valid Timing Difference Data\nAcross Any Learning Rates', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=16,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        plot_type = "No Data Available"
    
    ax3.set_xlabel('Learning Rate', fontsize=12)
    ax3.set_ylabel('Emergence Timing Difference (hours)', fontsize=12)
    ax3.set_title(f'Distribution: predicted_time - observed_time ({plot_type})', fontsize=14, pad=20)
    
    # Set x-axis ticks and labels
    ax3.set_xticks(range(len(learning_rates)))
    ax3.set_xticklabels([f'{lr:.4f}' for lr in learning_rates])
    
    # Customize the plot
    ax3.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax3.axhline(y=0, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Perfect Timing')
    
    # Add overall trend line using means
    valid_means = [stats['mean'] for stats in timing_stats if not np.isnan(stats['mean'])]
    valid_positions = [i for i, stats in enumerate(timing_stats) if not np.isnan(stats['mean'])]
    
    if len(valid_means) > 1:
        z = np.polyfit(valid_positions, valid_means, 1)
        p = np.poly1d(z)
        ax3.plot(valid_positions, p(valid_positions), "--", alpha=0.8, color='darkred', 
                linewidth=3, label='Mean Trend', zorder=7)
        
        # Add trend line formula to corner
        slope, intercept = z[0], z[1]
        formula = f'Trend: y = {slope:.4f}x + {intercept:.4f}'
        ax3.text(0.02, 0.98, formula, transform=ax3.transAxes, fontsize=10, 
                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', 
                facecolor='lightyellow', alpha=0.9, edgecolor='orange'))
    
    # Add legend
    handles, labels = ax3.get_legend_handles_labels()
    if handles:
        # Add custom legend entries for mean and std
        import matplotlib.patches as mpatches
        mean_patch = mpatches.Patch(color='black', label='Mean (◆)')
        std_patch = mpatches.Patch(color='gray', label='±1 Std Dev')
        
        # Combine all legend elements
        all_handles = handles + [mean_patch, std_patch]
        all_labels = labels + ['Mean (◆)', '±1 Std Dev']
        
        ax3.legend(all_handles, all_labels, loc='upper left', bbox_to_anchor=(1.02, 1), 
                  fontsize=10, framealpha=0.9)
    
    plt.tight_layout(rect=[0, 0.1, 0.85, 0.92])
    wandb.log({"timing_accuracy_comparison": wandb.Image(fig3)})
    plt.close(fig3)
    
    # 4. Combined Summary Plot - all key metrics in one view
    fig4, axes4 = plt.subplots(2, 3, figsize=(24, 16))  # Increased figure size significantly
    fig4.suptitle('Complete Performance Summary Across Learning Rates (3 Layers, 4 Heads)', fontsize=20, y=0.95)
    
    # List of all metrics to plot
    all_metrics = [
        ('overall_rmse', 'Overall RMSE', '#1f77b4'),
        ('overall_r2', 'Overall R²', '#ff7f0e'),
        ('emergence_rmse', 'Emergence RMSE', '#2ca02c'),
        ('emergence_r2', 'Emergence R²', '#d62728'),
        ('emergence_time_diff', 'Timing Error (hrs)', '#ff6b6b'),
        ('overall_mae', 'Overall MAE', '#9467bd')
    ]
    
    axes4_flat = axes4.flatten()
    
    for i, (metric, title, color) in enumerate(all_metrics):
        ax = axes4_flat[i]
        values = [all_results[lr][metric] for lr in learning_rates]
        
        # Create bar plot with adaptive width
        bars = ax.bar(range(len(learning_rates)), values, color=color, alpha=0.8, width=bar_width)
        
        # Add value labels on bars with minimal offset - directly on top
        max_val = max(values) if values else 1.0
        min_val = min(values) if values else 0.0
        val_range = max_val - min_val if max_val != min_val else abs(max_val) * 0.1
        text_offset = abs(max_val) * 0.02  # Much smaller offset - just 2% of max absolute value
        
        for j, (bar, value) in enumerate(zip(bars, values)):
            height = bar.get_height()
            if metric == 'emergence_time_diff':
                # Special handling for timing difference
                if height >= 0:
                    y_pos = height + text_offset
                    va = 'bottom'
                else:
                    y_pos = height - text_offset
                    va = 'top'
                ax.text(j, y_pos, f'{value:+.1f}', ha='center', va=va, fontsize=8, fontweight='bold')
            else:
                ax.text(j, height + text_offset, f'{value:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        ax.set_xlabel('Learning Rate', fontsize=11)
        ax.set_ylabel(title.split()[-1], fontsize=11)
        ax.set_title(title, fontsize=13, pad=15)
        
        # Set x-axis ticks and labels
        ax.set_xticks(range(len(learning_rates)))
        ax.set_xticklabels([f'{lr:.3f}' for lr in learning_rates], rotation=45, ha='right', fontsize=9)
        
        # Adjust y-axis limits to accommodate text with less space
        if metric == 'emergence_time_diff':
            max_abs_val = max(abs(min_val), abs(max_val)) if values else 1.0
            ax.set_ylim([-max_abs_val * 1.2, max_abs_val * 1.2])  # Reduced range
        else:
            ax.set_ylim([min_val - val_range * 0.1, max_val + val_range * 0.15])  # Reduced top margin
        
        ax.grid(True, alpha=0.3)
        
        # Add trend line
        if len(values) > 1:  # Only add trend line if we have multiple values
            z = np.polyfit(range(len(learning_rates)), values, 1)
            p = np.poly1d(z)
            ax.plot(range(len(learning_rates)), p(range(len(learning_rates))), "--", alpha=0.7, color='red', linewidth=1.5)
        
        # Add trend line formula to corner
        slope, intercept = z[0], z[1]
        formula = f'y = {slope:.4f}x + {intercept:.4f}'
        ax.text(0.02, 0.98, formula, transform=ax.transAxes, fontsize=8, 
                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.93])  # Better spacing
    wandb.log({"complete_performance_summary": wandb.Image(fig4)})
    plt.close(fig4)
    
    # Create summary table with key metrics
    summary_data = []
    for lr in learning_rates:
        row = {
            'learning_rate': lr,
            'overall_rmse': all_results[lr]['overall_rmse'],
            'overall_r2': all_results[lr]['overall_r2'],
            'emergence_rmse': all_results[lr]['emergence_rmse'],
            'emergence_r2': all_results[lr]['emergence_r2'],
            'emergence_time_error': all_results[lr]['emergence_time_diff']
        }
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    wandb.log({"lr_comparison_summary": wandb.Table(dataframe=summary_df)})

def create_epoch_mse_statistics_plot(all_results: Dict[float, Dict], config: Dict) -> None:
    """Create visualization showing mean±std of MSE across all epochs for each learning rate."""
    
    learning_rates = sorted(all_results.keys())
    
    # Calculate statistics for each learning rate
    emergence_stats = []
    overall_stats = []
    
    for lr in learning_rates:
        results = all_results[lr]
        
        # Emergence MSE statistics across epochs
        emergence_mses = results.get('test_emergence_mses', [])
        valid_emergence = [x for x in emergence_mses if not np.isnan(x)]
        
        if valid_emergence:
            emergence_stats.append({
                'mean': np.mean(valid_emergence),
                'std': np.std(valid_emergence),
                'min': np.min(valid_emergence),
                'max': np.max(valid_emergence),
                'count': len(valid_emergence)
            })
        else:
            emergence_stats.append({'mean': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan, 'count': 0})
        
        # Overall MSE statistics across epochs
        overall_mses = results.get('test_overall_mses', [])
        valid_overall = [x for x in overall_mses if not np.isnan(x)]
        
        if valid_overall:
            overall_stats.append({
                'mean': np.mean(valid_overall),
                'std': np.std(valid_overall),
                'min': np.min(valid_overall),
                'max': np.max(valid_overall),
                'count': len(valid_overall)
            })
        else:
            overall_stats.append({'mean': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan, 'count': 0})
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('MSE Statistics Across All Epochs for Each Learning Rate', fontsize=16, y=0.95)
    
    x_pos = range(len(learning_rates))
    bar_width = 0.6
    
    # Emergence MSE plot
    emergence_means = [stat['mean'] for stat in emergence_stats]
    emergence_stds = [stat['std'] for stat in emergence_stats]
    
    bars1 = ax1.bar(x_pos, emergence_means, yerr=emergence_stds, 
                   color='lightcoral', alpha=0.8, width=bar_width, capsize=5, 
                   label='Mean ± Std')
    
    # Add statistics text
    for i, (lr, stat) in enumerate(zip(learning_rates, emergence_stats)):
        if not np.isnan(stat['mean']):
            text = f"μ={stat['mean']:.4f}\nσ={stat['std']:.4f}\nmin={stat['min']:.4f}\nmax={stat['max']:.4f}\n(n={stat['count']})"
            ax1.text(i, stat['mean'] + stat['std'] + max(emergence_means) * 0.05, text, 
                    ha='center', va='bottom', fontsize=8, 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax1.set_title('Emergence MSE Across All Epochs', fontsize=14)
    ax1.set_xlabel('Learning Rate', fontsize=12)
    ax1.set_ylabel('Emergence MSE', fontsize=12)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'{lr:.3f}' for lr in learning_rates])
    ax1.grid(True, alpha=0.3)
    
    # Overall MSE plot
    overall_means = [stat['mean'] for stat in overall_stats]
    overall_stds = [stat['std'] for stat in overall_stats]
    
    bars2 = ax2.bar(x_pos, overall_means, yerr=overall_stds, 
                   color='lightblue', alpha=0.8, width=bar_width, capsize=5, 
                   label='Mean ± Std')
    
    # Add statistics text
    for i, (lr, stat) in enumerate(zip(learning_rates, overall_stats)):
        if not np.isnan(stat['mean']):
            text = f"μ={stat['mean']:.4f}\nσ={stat['std']:.4f}\nmin={stat['min']:.4f}\nmax={stat['max']:.4f}\n(n={stat['count']})"
            ax2.text(i, stat['mean'] + stat['std'] + max(overall_means) * 0.05, text, 
                    ha='center', va='bottom', fontsize=8, 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax2.set_title('Overall MSE Across All Epochs', fontsize=14)
    ax2.set_xlabel('Learning Rate', fontsize=12)
    ax2.set_ylabel('Overall MSE', fontsize=12)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'{lr:.3f}' for lr in learning_rates])
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.93])
    wandb.log({"epoch_mse_statistics": wandb.Image(fig)})
    plt.close(fig)
    
    # Also log the raw statistics as a table
    stats_data = []
    for i, lr in enumerate(learning_rates):
        stats_data.append({
            'learning_rate': lr,
            'emergence_mse_mean': emergence_stats[i]['mean'],
            'emergence_mse_std': emergence_stats[i]['std'],
            'emergence_mse_min': emergence_stats[i]['min'],
            'emergence_mse_max': emergence_stats[i]['max'],
            'overall_mse_mean': overall_stats[i]['mean'],
            'overall_mse_std': overall_stats[i]['std'],
            'overall_mse_min': overall_stats[i]['min'],
            'overall_mse_max': overall_stats[i]['max'],
            'epochs_count': emergence_stats[i]['count']
        })
    
    stats_df = pd.DataFrame(stats_data)
    wandb.log({"epoch_mse_statistics_table": wandb.Table(dataframe=stats_df)})

def create_per_trial_mse_statistics(results: Dict, learning_rate: float) -> None:
    """Create MSE statistics visualization for a single trial showing epoch-wise mean±std."""
    
    lr_str = f"{learning_rate:.4f}"
    
    # Extract MSE data across epochs
    emergence_mses = results.get('test_emergence_mses', [])
    overall_mses = results.get('test_overall_mses', [])
    
    # Filter out NaN values
    valid_emergence = [x for x in emergence_mses if not np.isnan(x)]
    valid_overall = [x for x in overall_mses if not np.isnan(x)]
    
    if not valid_emergence and not valid_overall:
        print(f"No valid MSE data for LR {learning_rate}")
        return
    
    # Calculate statistics
    emergence_stats = {
        'mean': np.mean(valid_emergence) if valid_emergence else np.nan,
        'std': np.std(valid_emergence) if valid_emergence else np.nan,
        'min': np.min(valid_emergence) if valid_emergence else np.nan,
        'max': np.max(valid_emergence) if valid_emergence else np.nan,
        'count': len(valid_emergence)
    }
    
    overall_stats = {
        'mean': np.mean(valid_overall) if valid_overall else np.nan,
        'std': np.std(valid_overall) if valid_overall else np.nan,
        'min': np.min(valid_overall) if valid_overall else np.nan,
        'max': np.max(valid_overall) if valid_overall else np.nan,
        'count': len(valid_overall)
    }
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'MSE Analysis for Learning Rate {learning_rate}', fontsize=16, y=0.95)
    
    # 1. Emergence MSE trajectory over epochs
    ax1 = axes[0]
    if valid_emergence:
        epochs = range(len(emergence_mses))
        ax1.plot(epochs, emergence_mses, 'lightcoral', alpha=0.7, linewidth=1, label='Epoch Values')
        ax1.axhline(y=emergence_stats['mean'], color='red', linestyle='-', linewidth=2, label=f"Mean = {emergence_stats['mean']:.4f}")
        ax1.axhline(y=emergence_stats['mean'] + emergence_stats['std'], color='red', linestyle='--', alpha=0.7, label=f"±1σ = {emergence_stats['std']:.4f}")
        ax1.axhline(y=emergence_stats['mean'] - emergence_stats['std'], color='red', linestyle='--', alpha=0.7)
    
    ax1.set_title('Emergence MSE Across Epochs', fontsize=12)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Emergence MSE')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Overall MSE trajectory over epochs  
    ax2 = axes[1]
    if valid_overall:
        epochs = range(len(overall_mses))
        ax2.plot(epochs, overall_mses, 'lightblue', alpha=0.7, linewidth=1, label='Epoch Values')
        ax2.axhline(y=overall_stats['mean'], color='blue', linestyle='-', linewidth=2, label=f"Mean = {overall_stats['mean']:.4f}")
        ax2.axhline(y=overall_stats['mean'] + overall_stats['std'], color='blue', linestyle='--', alpha=0.7, label=f"±1σ = {overall_stats['std']:.4f}")
        ax2.axhline(y=overall_stats['mean'] - overall_stats['std'], color='blue', linestyle='--', alpha=0.7)
    
    ax2.set_title('Overall MSE Across Epochs', fontsize=12)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Overall MSE')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add summary statistics text box
    stats_text = f"""Trial Statistics for LR {learning_rate}:

Emergence MSE:
  Mean = {emergence_stats['mean']:.4f}
  Std = {emergence_stats['std']:.4f}
  Min = {emergence_stats['min']:.4f}
  Max = {emergence_stats['max']:.4f}
  Epochs = {emergence_stats['count']}

Overall MSE:
  Mean = {overall_stats['mean']:.4f}
  Std = {overall_stats['std']:.4f}
  Min = {overall_stats['min']:.4f}
  Max = {overall_stats['max']:.4f}
  Epochs = {overall_stats['count']}"""
    
    fig.text(0.02, 0.02, stats_text, fontsize=10, verticalalignment='bottom',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout(rect=[0.25, 0.15, 1, 0.93])
    
    # Log to wandb under the specific learning rate tab
    wandb.log({f'lr_{lr_str}/mse_epoch_analysis': wandb.Image(fig)})
    plt.close(fig)
    
    # Also log individual statistics
    wandb.log({
        f'lr_{lr_str}/emergence_mse_mean_across_epochs': emergence_stats['mean'],
        f'lr_{lr_str}/emergence_mse_std_across_epochs': emergence_stats['std'],
        f'lr_{lr_str}/overall_mse_mean_across_epochs': overall_stats['mean'],
        f'lr_{lr_str}/overall_mse_std_across_epochs': overall_stats['std'],
    })

def main():
    print("Starting comprehensive learning rate comparison experiment...")
    
    # Read API key from .env file
    with open('/mmfs1/project/mx6/jst26/SAR_EMERGENCE_RESEARCH/.env', 'r') as f:
        for line in f:
            if line.startswith('WANDB_API_KEY='):
                api_key = line.strip().split('=')[1]
                break
    
    # Login to wandb first
    wandb.login(key=api_key)
    
    # Fixed configuration for all experiments
    config = {
        'num_pred': 12,
        'num_in': 110,
        'embed_dim': 64,
        'ff_dim': 128,
        'num_layers': 3,  # Fixed to 3 layers
        'dropout': 0.1, # play around, try 0.1-0.5
        'n_epochs': 300,
        'time_window': 12,
        'rid_of_top': 4,
        # learning_rate will be varied in the experiment
    }
    
    # Create models directory for saving best models
    # Save to results directory with search-specific subfolder
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    results_dir = os.path.join(project_root, 'transformer', 'results')
    search_type = "lr_search"  # This is a learning rate search experiment
    models_dir = os.path.join(results_dir, search_type)
    os.makedirs(models_dir, exist_ok=True)
    print(f"Models will be saved to: {models_dir}")
    
    # Initialize wandb run
    wandb.init(
        project="sar-emergence",
        entity="jonastirona-new-jersey-institute-of-technology",
        config=config,
        name="dropout 0.1 LR search",
        notes="Comprehensive grid search comparing transformer performance across smaller learning rates (0.0001, 0.0005, 0.001, 0.005, 0.01) with constant learning rate schedule, fixed attention head count (4) and 3 layers"
    )
    
    # Run experiments for different learning rates
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_results = {}
    
    learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]  # Much smaller learning rates
    global_step_counter = 0  # Track global step across all trials
    
    for lr in learning_rates:
        print(f"\n{'='*50}")
        print(f"TRIAL {learning_rates.index(lr) + 1}/5: Testing learning rate {lr}")
        print(f"{'='*50}")
        
        try:
            # Run experiment with global step offset
            results = run_single_experiment(config, device, lr, global_step_counter)
            all_results[lr] = results
            
            # Create per-trial MSE statistics visualization
            print(f"Creating MSE statistics for LR {lr}...")
            create_per_trial_mse_statistics(results, lr)
            
            # Save the best model locally
            model_filename = f"transformer_t{config['time_window']}_r{config['rid_of_top']}_i{config['num_in']}_n{config['num_layers']}_h{config['embed_dim']}_e{config['n_epochs']}_l{lr:.5f}_d{config['dropout']:.1f}.pth"
            model_path = os.path.join(models_dir, model_filename)
            torch.save(results['best_model_state'], model_path)
            print(f"Best model saved to: {model_path}")
            
            # Save model to wandb as artifact
            model_artifact = wandb.Artifact(
                name=f"transformer_t{config['time_window']}_r{config['rid_of_top']}_i{config['num_in']}_n{config['num_layers']}_h{config['embed_dim']}_e{config['n_epochs']}_l{lr:.5f}_d{config['dropout']:.1f}".replace('.', '_'),
                type="model",
                description=f"Best transformer model for learning rate {lr} after {config['n_epochs']} epochs",
                metadata={
                    "learning_rate": lr,
                    "epochs": config['n_epochs'],
                    "embed_dim": config['embed_dim'],
                    "num_heads": 4,
                    "ff_dim": config['ff_dim'],
                    "num_layers": config['num_layers'],
                    "dropout": config['dropout'],
                    "time_window": config['time_window'],
                    "rid_of_top": config['rid_of_top'],
                    "num_in": config['num_in'],
                    "test_loss": results.get('overall_rmse', 'N/A'),
                    "emergence_rmse": results.get('emergence_rmse', 'N/A'),
                    "model_type": "SpatioTemporalTransformer"
                }
            )
            model_artifact.add_file(model_path, name=model_filename)
            wandb.log_artifact(model_artifact)
            print(f"Model uploaded to wandb as artifact: {model_artifact.name}")
            
            # Update global step counter for next trial (assuming 300 epochs per trial)
            global_step_counter += config['n_epochs']
            
            # Log trial completion
            wandb.log({
                f'trial_completion/lr_{lr:.3f}': 1.0,
                'completed_trials': learning_rates.index(lr) + 1,
                'total_trials': len(learning_rates),
                'model_saved_path': model_path
            }, step=global_step_counter - 1)
            
            print(f"✓ Trial {learning_rates.index(lr) + 1}/5 completed successfully for learning rate {lr}")
            
        except Exception as e:
            print(f"✗ ERROR in Trial {learning_rates.index(lr) + 1}/5 for learning rate {lr}:")
            print(f"  Exception: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Log the failure
            wandb.log({
                f'trial_completion/lr_{lr:.3f}': 0.0,
                'completed_trials': learning_rates.index(lr) + 1,
                'total_trials': len(learning_rates),
                'trial_failed': True,
                'error_message': str(e)
            }, step=global_step_counter + config['n_epochs'] - 1)
            
            global_step_counter += config['n_epochs']
            continue
        
        # Create individual AR evaluation plots using eval module
        print(f"Generating AR comparison plots with LSTM benchmark...")
        
        # LSTM model path (assuming it exists)
        lstm_path = "/mmfs1/project/mx6/jst26/SAR_EMERGENCE_RESEARCH/lstm/results/t12_r4_i110_n3_h64_e1000_l0.01.pth"
        
        transformer_params = {
            'embed_dim': config['embed_dim'],
            'num_heads': 4,  # Fixed attention head count
            'ff_dim': config['ff_dim'],
            'num_layers': config['num_layers'],  # Fixed to 3 layers
            'dropout': config['dropout'],
            'rid_of_top': config['rid_of_top'],
            'num_pred': config['num_pred'],
            'time_window': config['time_window'],
            'num_in': config['num_in'],
            'hidden_size': config['embed_dim'],  # Use embed_dim as hidden_size
            'learning_rate': lr
        }
        
        # Test ARs to evaluate
        test_ars = [11698, 11726, 13165, 13179, 13183]
        successful_ars = []
        failed_ars = []
        
        lr_str = f"{lr:.4f}"
        
        for ar in test_ars:
            try:
                # Create a temporary output directory for this evaluation
                temp_output_dir = f"/tmp/ar_eval_lr{lr:.3f}"
                os.makedirs(temp_output_dir, exist_ok=True)
                plot_path = evaluate_models_for_ar(ar, lstm_path, model_path, transformer_params, temp_output_dir)
                
                # Upload the generated plot to wandb
                if plot_path and os.path.exists(plot_path):
                    wandb.log({f'lr_{lr_str}/AR_{ar}_comparison': wandb.Image(plot_path)})
                    successful_ars.append(ar)
                else:
                    failed_ars.append(ar)
                    
            except Exception as e:
                failed_ars.append(ar)
                print(f"  ✗ Error evaluating AR {ar}: {str(e)}")
                continue
        
        print(f"Completed AR evaluations for learning rate {lr}: {len(successful_ars)}/{len(test_ars)} successful")
        
        # Create artifact with all evaluation plots for this learning rate
        if successful_ars:
            eval_plots_artifact = wandb.Artifact(
                name=f"evaluation_plots_lr_{lr:.3f}".replace('.', '_'),
                type="evaluation_plots",
                description=f"AR comparison plots for learning rate {lr}",
                metadata={
                    "learning_rate": lr,
                    "successful_ars": successful_ars,
                    "failed_ars": failed_ars,
                    "total_ars_attempted": len(test_ars),
                    "success_rate": len(successful_ars) / len(test_ars)
                }
            )
            
            # Add successful evaluation plots to artifact
            for ar in successful_ars:
                plot_filename = f"AR{ar}_comparison.png"
                plot_path = f"/tmp/ar_eval_lr{lr:.3f}/{plot_filename}"
                if os.path.exists(plot_path):
                    eval_plots_artifact.add_file(plot_path, name=plot_filename)
            
            wandb.log_artifact(eval_plots_artifact)
            print(f"Evaluation plots artifact created: {eval_plots_artifact.name}")
        
        print(f"✓ Trial {learning_rates.index(lr) + 1}/5 fully completed including AR evaluations")
    
    # Create comprehensive comparison plots
    print(f"\n{'='*50}")
    print("Creating comprehensive comparison plots...")
    print(f"{'='*50}")
    create_cross_trial_comparison_plots(all_results, config)
    
    # Create epoch-wise MSE statistics visualization
    print("Creating epoch-wise MSE statistics...")
    create_epoch_mse_statistics_plot(all_results, config)
    
    # Log comprehensive summary of all trials
    print("Logging comprehensive trial summary...")
    
    # Create a summary table with all trial results
    trial_summary_data = []
    for lr in sorted(all_results.keys()):
        results = all_results[lr]
        trial_data = {
            'learning_rate': lr,
            'train_loss_final': results['train_losses'][-1] if results['train_losses'] else float('nan'),
            'test_loss_final': results['test_losses'][-1] if results['test_losses'] else float('nan'),
            'test_overall_rmse': results.get('overall_rmse', float('nan')),
            'test_overall_r2': results.get('overall_r2', float('nan')),
            'test_overall_mse': results.get('overall_mse', float('nan')),
            'test_overall_mae': results.get('overall_mae', float('nan')),
            'test_emergence_rmse': results.get('emergence_rmse', float('nan')),
            'test_emergence_r2': results.get('emergence_r2', float('nan')),
            'test_emergence_mse': results.get('emergence_mse', float('nan')),
            'test_emergence_mae': results.get('emergence_mae', float('nan')),
            'emergence_time_diff': results.get('emergence_time_diff', float('nan')),
            'emergence_window_size': results.get('emergence_window_size', float('nan')),
            'num_valid_tiles': results.get('num_valid_tiles', 0),
            'tile_success_rate': results.get('tile_success_rate', 0.0)
        }
        trial_summary_data.append(trial_data)
        
        # Also log individual trial summary metrics with final step
        final_step = (learning_rates.index(lr) + 1) * config['n_epochs'] - 1  # Correct step for each trial
        wandb.log({
            f'final_results/lr_{lr:.4f}_train_loss': trial_data['train_loss_final'],
            f'final_results/lr_{lr:.4f}_test_loss': trial_data['test_loss_final'],
            f'final_results/lr_{lr:.4f}_test_overall_rmse': trial_data['test_overall_rmse'],
            f'final_results/lr_{lr:.4f}_test_overall_r2': trial_data['test_overall_r2'],
            f'final_results/lr_{lr:.4f}_test_overall_mse': trial_data['test_overall_mse'],
            f'final_results/lr_{lr:.4f}_test_overall_mae': trial_data['test_overall_mae'],
            f'final_results/lr_{lr:.4f}_test_emergence_rmse': trial_data['test_emergence_rmse'],
            f'final_results/lr_{lr:.4f}_test_emergence_r2': trial_data['test_emergence_r2'],
            f'final_results/lr_{lr:.4f}_test_emergence_mse': trial_data['test_emergence_mse'],
            f'final_results/lr_{lr:.4f}_test_emergence_mae': trial_data['test_emergence_mae'],
            f'final_results/lr_{lr:.4f}_emergence_time_diff': trial_data['emergence_time_diff']
        }, step=final_step)
    
    # Log the comprehensive trial summary table
    trial_summary_df = pd.DataFrame(trial_summary_data)
    wandb.log({"comprehensive_trial_summary": wandb.Table(dataframe=trial_summary_df)})
    
    # Create summary artifact with all models
    print("Creating wandb artifact with all models...")
    all_models_artifact = wandb.Artifact(
        name="lr_search_all_models",
        type="model_collection",
        description=f"All transformer models from learning rate search experiment with {len(learning_rates)} learning rates",
        metadata={
            "experiment_type": "learning_rate_search",
            "learning_rates": learning_rates,
            "n_epochs": config['n_epochs'],
            "embed_dim": config['embed_dim'],
            "num_heads": 4,
            "ff_dim": config['ff_dim'],
            "num_layers": config['num_layers'],
            "dropout": config['dropout'],
            "best_overall_lr": min(all_results.keys(), key=lambda x: all_results[x].get('overall_rmse', float('inf'))),
            "best_emergence_lr": min(all_results.keys(), key=lambda x: all_results[x].get('emergence_rmse', float('inf'))),
            "best_timing_lr": min(all_results.keys(), key=lambda x: all_results[x].get('emergence_time_diff', float('inf'))),
            "total_models": len(all_results)
        }
    )
    
    # Add all model files to the collection artifact
    for lr in learning_rates:
        if lr in all_results:  # Only add if training was successful
            model_filename = f"transformer_t{config['time_window']}_r{config['rid_of_top']}_i{config['num_in']}_n{config['num_layers']}_h{config['embed_dim']}_e{config['n_epochs']}_l{lr:.5f}_d{config['dropout']:.1f}.pth"
            model_path = os.path.join(models_dir, model_filename)
            if os.path.exists(model_path):
                all_models_artifact.add_file(model_path, name=model_filename)
    
    wandb.log_artifact(all_models_artifact)
    print(f"All models collection uploaded to wandb as artifact: {all_models_artifact.name}")
    
    # Log summary statistics
    summary_stats = {
        'total_trials_completed': len(all_results),
        'best_overall_rmse_lr': min(all_results.keys(), key=lambda x: all_results[x].get('overall_rmse', float('inf'))),
        'best_emergence_rmse_lr': min(all_results.keys(), key=lambda x: all_results[x].get('emergence_rmse', float('inf'))),
        'best_timing_accuracy_lr': min(all_results.keys(), key=lambda x: all_results[x].get('emergence_time_diff', float('inf'))),
        'all_trials_rmse_range': max([r.get('overall_rmse', 0) for r in all_results.values()]) - min([r.get('overall_rmse', 0) for r in all_results.values()]),
        'avg_emergence_rmse': np.mean([r.get('emergence_rmse', 0) for r in all_results.values()]),
        'avg_overall_rmse': np.mean([r.get('overall_rmse', 0) for r in all_results.values()])
    }
    wandb.log(summary_stats)
    
    # Find and log best configurations
    best_overall = min(all_results.keys(), key=lambda x: all_results[x]['overall_rmse'])
    best_emergence = min(all_results.keys(), key=lambda x: all_results[x]['emergence_rmse'])
    best_time_error = min(all_results.keys(), key=lambda x: all_results[x]['emergence_time_diff'])
    
    # Log best configurations as simple text values
    wandb.log({
        'best_overall_lr': best_overall,
        'best_emergence_lr': best_emergence,
        'best_timing_lr': best_time_error,
        'best_overall_rmse': all_results[best_overall]['overall_rmse'],
        'best_emergence_rmse': all_results[best_emergence]['emergence_rmse'],
        'best_time_error': all_results[best_time_error]['emergence_time_diff']
    })
    
    # Also log as summary text for easy viewing
    summary_text = f"""
    BEST CONFIGURATIONS:
    - Overall Performance: LR={best_overall} (RMSE: {all_results[best_overall]['overall_rmse']:.4f})
    - Emergence Prediction: LR={best_emergence} (RMSE: {all_results[best_emergence]['emergence_rmse']:.4f})  
    - Timing Accuracy: LR={best_time_error} (Error: {all_results[best_time_error]['emergence_time_diff']:.2f} hrs)
    """
    wandb.log({"best_configurations_summary": summary_text})
    
    print(f"\n{'='*50}")
    print("EXPERIMENT COMPLETE!")
    print(f"Best overall performance: LR={best_overall} (RMSE: {all_results[best_overall]['overall_rmse']:.4f})")
    print(f"Best emergence prediction: LR={best_emergence} (RMSE: {all_results[best_emergence]['emergence_rmse']:.4f})")
    print(f"Best timing accuracy: LR={best_time_error} (Error: {all_results[best_time_error]['emergence_time_diff']:.2f} hrs)")
    print(f"{'='*50}")
    
    # Print model saving summary
    print(f"\n{'='*50}")
    print("SAVED MODELS SUMMARY")
    print(f"{'='*50}")
    print(f"Local directory: {models_dir}")
    print(f"Search type: {search_type}")
    print(f"Wandb artifacts: Individual models + collection artifact")
    print("\nSaved models:")
    for lr in learning_rates:
        model_filename = f"transformer_t{config['time_window']}_r{config['rid_of_top']}_i{config['num_in']}_n{config['num_layers']}_h{config['embed_dim']}_e{config['n_epochs']}_l{lr:.5f}_d{config['dropout']:.1f}.pth"
        model_path = os.path.join(models_dir, model_filename)
        artifact_name = f"transformer_t{config['time_window']}_r{config['rid_of_top']}_i{config['num_in']}_n{config['num_layers']}_h{config['embed_dim']}_e{config['n_epochs']}_l{lr:.5f}_d{config['dropout']:.1f}".replace('.', '_')
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path) / (1024*1024)  # Size in MB
            print(f"  ✓ LR={lr}: {model_filename} ({file_size:.1f} MB)")
            print(f"    Wandb artifact: {artifact_name}")
        else:
            print(f"  ✗ LR={lr}: {model_filename} (MISSING)")
    
    print(f"\nWandb artifact collection: lr_search_all_models")
    print(f"  Contains: {len([lr for lr in learning_rates if lr in all_results])} models")
    print(f"\nTo download models from wandb:")
    print(f"  # Download individual model")
    print(f"  artifact = wandb.use_artifact('your-entity/sar-emergence/transformer_t12_r4_i110_n3_h64_e300_l0_00100_d0_3:latest')")
    print(f"  artifact_dir = artifact.download()")
    print(f"  # Download all models")
    print(f"  artifact = wandb.use_artifact('your-entity/sar-emergence/lr_search_all_models:latest')")
    print(f"  artifact_dir = artifact.download()")
    
    print(f"\nTo load a model:")
    print(f"  model_state = torch.load('{models_dir}/transformer_t{config['time_window']}_r{config['rid_of_top']}_i{config['num_in']}_n{config['num_layers']}_h{config['embed_dim']}_e{config['n_epochs']}_l{lr:.5f}_d{config['dropout']:.1f}.pth')")
    print(f"{'='*50}")
    
    # Finish wandb run
    wandb.finish()

def lstm_ready(tile, size, power_maps, intensities, num_in, num_pred):
    """Use the same data preprocessing as LSTM code.
    
    Args:
        tile: Tile index
        size: Grid size
        power_maps: Power maps data
        intensities: Intensity data
        num_in: Requested input sequence length
        num_pred: Number of prediction steps
    """
    # Use the same preprocessing as LSTM code
    final_maps = np.transpose(power_maps, axes=(2, 1, 0))  # (time, features, tiles)
    final_ints = np.transpose(intensities, axes=(1,0))     # (time, tiles)
    X_trans = final_maps[:,:,tile]  # (time, features)
    y_trans = final_ints[:,tile]    # (time,)
    
    # Split into sequences
    X_ss, y_mm = split_sequences(X_trans, y_trans, num_in, num_pred)
    
    # Convert to tensors
    X = torch.Tensor(X_ss)  # (batch, seq_len, input_dim)
    y = torch.Tensor(y_mm)  # (batch, output_dim)
    
    return X, y

if __name__ == "__main__":
    main()