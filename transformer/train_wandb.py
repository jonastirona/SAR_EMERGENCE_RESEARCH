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
from transformer.eval_wandb import evaluate_models_for_ar

# Set up logging
logging.basicConfig(
    level=logging.WARNING,  # Changed from INFO to WARNING
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Cosine annealing scheduler with linear warmup"""
    def __init__(self, optimizer, warmup_epochs, max_epochs, eta_min=1e-6, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs 
                    for base_lr in self.base_lrs]
        else:
            # Cosine annealing after warmup
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            return [self.eta_min + (base_lr - self.eta_min) * 0.5 * (1 + math.cos(math.pi * progress))
                    for base_lr in self.base_lrs]

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
    
    emergence_time_diff = abs(obs_emergence_start - pred_emergence_start) if pred_emergence_start is not None else None
    
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

def run_single_experiment(config: Dict[str, Any], device: torch.device, num_layers: int, global_step_offset: int = 0) -> Dict[str, float]:
    """Run a single experiment with the given configuration."""
    print(f"Training model with {num_layers} layers...")
    
    # Initialize model
    model = SpatioTemporalTransformer(
        input_dim=5,  # 4 power maps + 1 magnetic flux
        seq_len=config['num_in'],
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        num_layers=num_layers,
        ff_dim=config['ff_dim'],
        dropout=config['dropout'],
        output_dim=config['num_pred']
    ).to(device)
    
    # ARs list (copied from train_w_stats.py)
    ARs = [11130,11149,11158,11162,11199,11327,11344,11387,11393,11416,11422,11455,11619,11640,11660,11678,11682,11765,11768,11776,11916,11928,12036,12051,12085,12089,12144,12175,12203,12257,12331,12494,12659,12778,12864,12877,12900,12929,13004,13085,13098,13179]
    size = 9
    rid_of_top = config['rid_of_top']
    num_in = config['num_in']
    num_pred = config['num_pred']

    all_inputs, all_intensities = load_all_ars_data(ARs, rid_of_top, size, num_in, num_pred)
    input_size = np.shape(all_inputs)[1]

    # Prepare data for all tiles and ARs - keep track of tile indices
    X_trains = []
    y_trains = []
    tile_indices = []  # Track which tile each sample comes from
    remaining_rows = size - 2*rid_of_top
    tiles = remaining_rows * size
    
    for ar_idx in range(len(ARs)):
        power_maps = all_inputs[:,:,:,ar_idx]
        intensities = all_intensities[:,:,ar_idx]
        for tile in range(tiles):
            X_tile, y_tile = lstm_ready(tile, size, power_maps, intensities, num_in, num_pred, model_seq_len=num_in)
            X_trains.append(X_tile)
            y_trains.append(y_tile)
            # Add tile indices for each sequence in this tile
            tile_indices.extend([tile] * len(X_tile))
    
    X = torch.cat(X_trains, dim=0)
    y = torch.cat(y_trains, dim=0)
    X = torch.reshape(X, (X.shape[0], num_in, X.shape[2]))
    tile_indices = np.array(tile_indices)

    # Split data first on CPU
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    tile_indices_train, tile_indices_test = tile_indices[:train_size], tile_indices[train_size:]
    
    # Create DataLoaders with CPU data
    batch_size = min(64, len(X_train) // 10)  # Adaptive batch size
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
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = CosineWarmupScheduler(optimizer, warmup_epochs=5, max_epochs=config['n_epochs'])
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
    
    for epoch in range(config['n_epochs']):
        # Training phase
        model.train()
        epoch_train_loss = 0
        num_batches = 0
        
        for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
            # Move batch to device
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
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
            
            # Calculate unique global step for this trial and epoch
            global_step = global_step_offset + epoch
            
            # Log per-epoch metrics with layer prefix and unique global step
            metrics_to_log = {
                f'layers_{num_layers}/train_loss': epoch_train_loss,
                f'layers_{num_layers}/test_loss': epoch_test_loss,
                f'layers_{num_layers}/test_emergence_rmse': test_emergence_metrics['emergence_rmse'],
                f'layers_{num_layers}/test_emergence_mse': test_emergence_metrics.get('emergence_mse', float('nan')),
                f'layers_{num_layers}/test_emergence_mae': test_emergence_metrics.get('emergence_mae', float('nan')),
                f'layers_{num_layers}/test_emergence_r2': test_emergence_metrics.get('emergence_r2', float('nan')),
                f'layers_{num_layers}/test_overall_rmse': test_emergence_metrics.get('overall_rmse', float('nan')),
                f'layers_{num_layers}/test_overall_mse': test_emergence_metrics.get('overall_mse', float('nan')),
                f'layers_{num_layers}/test_overall_mae': test_emergence_metrics.get('overall_mae', float('nan')),
                f'layers_{num_layers}/test_overall_r2': test_emergence_metrics.get('overall_r2', float('nan')),
                f'layers_{num_layers}/learning_rate': optimizer.param_groups[0]['lr'],
                f'layers_{num_layers}/epoch': epoch,
                # Also log without layer prefix for easy cross-trial comparison
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
                'num_layers': num_layers,  # Add layer info for filtering
                'trial_id': f'layers_{num_layers}'  # Add trial identifier
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
    
    # Add layer info to metrics
    final_metrics['num_layers'] = num_layers
    
    # Return all metrics including the lists for plotting and best model state
    return {
        **final_metrics,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_emergence_losses': test_emergence_rmses,  # Use test for consistency
        'test_emergence_losses': test_emergence_rmses,
        'best_model_state': best_model_state
    }

def create_cross_trial_comparison_plots(all_results: Dict[int, Dict], config: Dict) -> None:
    """Create comprehensive comparison plots across all layer trials with separate grids for each metric."""
    
    # Extract data for plotting
    layer_counts = sorted(all_results.keys())
    
    # Create two comprehensive figures - one for overall metrics, one for emergence metrics
    
    # 1. Overall Performance Metrics - 2x2 grid
    fig1, axes1 = plt.subplots(2, 2, figsize=(16, 12))
    fig1.suptitle('Overall Performance Metrics Across Transformer Layers', fontsize=18, y=0.95)
    
    overall_metrics = ['overall_mse', 'overall_rmse', 'overall_mae', 'overall_r2']
    metric_labels = ['MSE', 'RMSE', 'MAE', 'R²']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Flatten axes for easier iteration
    axes1_flat = axes1.flatten()
    
    for i, (metric, label, color) in enumerate(zip(overall_metrics, metric_labels, colors)):
        ax = axes1_flat[i]
        values = [all_results[layers][metric] for layers in layer_counts]
        
        # Create bar plot
        bars = ax.bar(layer_counts, values, color=color, alpha=0.8, width=0.6)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.02,
                    f'{value:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Number of Transformer Layers', fontsize=12)
        ax.set_ylabel(f'{label} Value', fontsize=12)
        ax.set_title(f'Overall {label}', fontsize=14, pad=15)
        ax.set_xticks(layer_counts)
        ax.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(layer_counts, values, 1)
        p = np.poly1d(z)
        ax.plot(layer_counts, p(layer_counts), "--", alpha=0.7, color='red', linewidth=2)
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    wandb.log({"overall_metrics_comparison": wandb.Image(fig1)})
    plt.close(fig1)
    
    # 2. Emergence Window Performance Metrics - 2x2 grid  
    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
    fig2.suptitle('Emergence Window Performance Metrics (±12 hrs) Across Transformer Layers', fontsize=18, y=0.95)
    
    emergence_metrics = ['emergence_mse', 'emergence_rmse', 'emergence_mae', 'emergence_r2']
    
    # Flatten axes for easier iteration
    axes2_flat = axes2.flatten()
    
    for i, (metric, label, color) in enumerate(zip(emergence_metrics, metric_labels, colors)):
        ax = axes2_flat[i]
        values = [all_results[layers][metric] for layers in layer_counts]
        
        # Create bar plot
        bars = ax.bar(layer_counts, values, color=color, alpha=0.8, width=0.6)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.02,
                    f'{value:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Number of Transformer Layers', fontsize=12)
        ax.set_ylabel(f'{label} Value', fontsize=12)
        ax.set_title(f'Emergence Window {label}', fontsize=14, pad=15)
        ax.set_xticks(layer_counts)
        ax.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(layer_counts, values, 1)
        p = np.poly1d(z)
        ax.plot(layer_counts, p(layer_counts), "--", alpha=0.7, color='red', linewidth=2)
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    wandb.log({"emergence_metrics_comparison": wandb.Image(fig2)})
    plt.close(fig2)
    
    # 3. Emergence Timing Accuracy - single plot
    fig3, ax3 = plt.subplots(1, 1, figsize=(10, 6))
    fig3.suptitle('Emergence Timing Accuracy Across Transformer Layers', fontsize=16, y=0.95)
    
    time_errors = [all_results[layers]['emergence_time_diff'] for layers in layer_counts]
    bars = ax3.bar(layer_counts, time_errors, color='#ff6b6b', alpha=0.8, width=0.6)
    
    # Add value labels on bars
    for bar, value in zip(bars, time_errors):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + max(time_errors)*0.02,
                f'{value:.1f}h', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax3.set_xlabel('Number of Transformer Layers', fontsize=12)
    ax3.set_ylabel('Time Error (hours)', fontsize=12)
    ax3.set_title('|predicted_time - observed_time|', fontsize=14, pad=15)
    ax3.set_xticks(layer_counts)
    ax3.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(layer_counts, time_errors, 1)
    p = np.poly1d(z)
    ax3.plot(layer_counts, p(layer_counts), "--", alpha=0.7, color='red', linewidth=2)
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    wandb.log({"timing_accuracy_comparison": wandb.Image(fig3)})
    plt.close(fig3)
    
    # 4. Combined Summary Plot - all key metrics in one view
    fig4, axes4 = plt.subplots(2, 3, figsize=(20, 12))
    fig4.suptitle('Complete Performance Summary Across Transformer Layers', fontsize=18, y=0.95)
    
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
        values = [all_results[layers][metric] for layers in layer_counts]
        
        # Create bar plot
        bars = ax.bar(layer_counts, values, color=color, alpha=0.8, width=0.6)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.02,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Layers', fontsize=10)
        ax.set_ylabel(title.split()[-1], fontsize=10)
        ax.set_title(title, fontsize=12, pad=10)
        ax.set_xticks(layer_counts)
        ax.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(layer_counts, values, 1)
        p = np.poly1d(z)
        ax.plot(layer_counts, p(layer_counts), "--", alpha=0.7, color='red', linewidth=1.5)
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    wandb.log({"complete_performance_summary": wandb.Image(fig4)})
    plt.close(fig4)
    
    # Create summary table with key metrics
    summary_data = []
    for layers in layer_counts:
        row = {
            'layers': layers,
            'overall_rmse': all_results[layers]['overall_rmse'],
            'overall_r2': all_results[layers]['overall_r2'],
            'emergence_rmse': all_results[layers]['emergence_rmse'],
            'emergence_r2': all_results[layers]['emergence_r2'],
            'emergence_time_error': all_results[layers]['emergence_time_diff']
        }
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    wandb.log({"layer_comparison_summary": wandb.Table(dataframe=summary_df)})

def main():
    print("Starting comprehensive layer comparison experiment...")
    
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
        'num_heads': 8,
        'ff_dim': 128,
        'dropout': 0.1,
        'learning_rate': 0.001,
        'n_epochs': 100,
        'time_window': 12,
        'rid_of_top': 4
    }
    
    # Initialize wandb run
    wandb.init(
        project="sar-emergence",
        entity="jonastirona-new-jersey-institute-of-technology",
        config=config,
        name="Layer Search",
        notes="Comprehensive grid search comparing transformer performance across 1-5 layers with AR emergence evaluation"
    )
    
    # Run experiments for 1-5 layers
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_results = {}
    
    layer_range = range(1, 6)  # 1 to 5 layers
    global_step_counter = 0  # Track global step across all trials
    
    for num_layers in layer_range:
        print(f"\n{'='*50}")
        print(f"TRIAL {num_layers}/5: Testing {num_layers} layer(s)")
        print(f"{'='*50}")
        
        # Run experiment with global step offset
        results = run_single_experiment(config, device, num_layers, global_step_counter)
        all_results[num_layers] = results
        
        # Update global step counter for next trial (assuming 10 epochs per trial)
        global_step_counter += config['n_epochs']
        
        # Log trial completion
        wandb.log({
            f'trial_completion/layers_{num_layers}': 1.0,
            'completed_trials': num_layers,
            'total_trials': len(layer_range)
        }, step=global_step_counter - 1)
        
        # Create individual AR evaluation plots using the new eval_wandb module
        print(f"Generating AR comparison plots with LSTM benchmark...")
        
        # We need to save the model temporarily to use with the evaluation function
        temp_model_path = f"/tmp/transformer_{num_layers}layers.pth"
        torch.save(results['best_model_state'], temp_model_path)
        
        # LSTM model path (assuming it exists)
        lstm_path = "/mmfs1/project/mx6/jst26/SAR_EMERGENCE_RESEARCH/lstm/results/t12_r4_i110_n3_h64_e1000_l0.01.pth"
        
        transformer_params = {
            'embed_dim': config['embed_dim'],
            'num_heads': config['num_heads'],
            'ff_dim': config['ff_dim'],
            'num_layers': num_layers,
            'dropout': config['dropout'],
            'rid_of_top': config['rid_of_top'],
            'num_pred': config['num_pred'],
            'time_window': config['time_window'],
            'num_in': config['num_in'],
            'hidden_size': config['embed_dim'],  # Use embed_dim as hidden_size
            'learning_rate': config['learning_rate']
        }
        
        # Test ARs to evaluate
        test_ars = [11698, 11726, 13165, 13179, 13183]
        successful_ars = []
        failed_ars = []
        
        for ar in test_ars:
            try:
                # Create a temporary output directory for this evaluation
                temp_output_dir = f"/tmp/ar_eval_{num_layers}layers"
                os.makedirs(temp_output_dir, exist_ok=True)
                evaluate_models_for_ar(ar, lstm_path, temp_model_path, transformer_params, temp_output_dir)
                
                # Upload the generated plot to wandb
                plot_path = os.path.join(temp_output_dir, f"AR{ar}_comparison.png")
                if os.path.exists(plot_path):
                    # Load the image and log to wandb
                    wandb.log({f'layers_{num_layers}/AR_{ar}_comparison': wandb.Image(plot_path)})
                    successful_ars.append(ar)
                else:
                    failed_ars.append(ar)
                    
            except Exception as e:
                failed_ars.append(ar)
                print(f"  ✗ Error evaluating AR {ar}: {str(e)}")
                continue
        
        # Clean up temporary file
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)
            
        print(f"Completed AR evaluations for {num_layers} layers: {len(successful_ars)}/{len(test_ars)} successful")
    
    # Create comprehensive comparison plots
    print(f"\n{'='*50}")
    print("Creating comprehensive comparison plots...")
    print(f"{'='*50}")
    create_cross_trial_comparison_plots(all_results, config)
    
    # Log comprehensive summary of all trials
    print("Logging comprehensive trial summary...")
    
    # Create a summary table with all trial results
    trial_summary_data = []
    for layers in sorted(all_results.keys()):
        results = all_results[layers]
        trial_data = {
            'num_layers': layers,
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
        final_step = global_step_counter + layers * 10  # Ensure unique final steps
        wandb.log({
            f'final_results/layers_{layers}_train_loss': trial_data['train_loss_final'],
            f'final_results/layers_{layers}_test_loss': trial_data['test_loss_final'],
            f'final_results/layers_{layers}_test_overall_rmse': trial_data['test_overall_rmse'],
            f'final_results/layers_{layers}_test_overall_r2': trial_data['test_overall_r2'],
            f'final_results/layers_{layers}_test_overall_mse': trial_data['test_overall_mse'],
            f'final_results/layers_{layers}_test_overall_mae': trial_data['test_overall_mae'],
            f'final_results/layers_{layers}_test_emergence_rmse': trial_data['test_emergence_rmse'],
            f'final_results/layers_{layers}_test_emergence_r2': trial_data['test_emergence_r2'],
            f'final_results/layers_{layers}_test_emergence_mse': trial_data['test_emergence_mse'],
            f'final_results/layers_{layers}_test_emergence_mae': trial_data['test_emergence_mae'],
            f'final_results/layers_{layers}_emergence_time_diff': trial_data['emergence_time_diff']
        }, step=final_step)
    
    # Log the comprehensive trial summary table
    trial_summary_df = pd.DataFrame(trial_summary_data)
    wandb.log({"comprehensive_trial_summary": wandb.Table(dataframe=trial_summary_df)})
    
    # Log summary statistics
    summary_stats = {
        'total_trials_completed': len(all_results),
        'best_overall_rmse_layers': min(all_results.keys(), key=lambda x: all_results[x].get('overall_rmse', float('inf'))),
        'best_emergence_rmse_layers': min(all_results.keys(), key=lambda x: all_results[x].get('emergence_rmse', float('inf'))),
        'best_timing_accuracy_layers': min(all_results.keys(), key=lambda x: all_results[x].get('emergence_time_diff', float('inf'))),
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
        'best_overall_layers': best_overall,
        'best_emergence_layers': best_emergence,
        'best_timing_layers': best_time_error,
        'best_overall_rmse': all_results[best_overall]['overall_rmse'],
        'best_emergence_rmse': all_results[best_emergence]['emergence_rmse'],
        'best_time_error': all_results[best_time_error]['emergence_time_diff']
    })
    
    # Also log as summary text for easy viewing
    summary_text = f"""
    BEST CONFIGURATIONS:
    - Overall Performance: {best_overall} layers (RMSE: {all_results[best_overall]['overall_rmse']:.4f})
    - Emergence Prediction: {best_emergence} layers (RMSE: {all_results[best_emergence]['emergence_rmse']:.4f})  
    - Timing Accuracy: {best_time_error} layers (Error: {all_results[best_time_error]['emergence_time_diff']:.2f} hrs)
    """
    wandb.log({"best_configurations_summary": summary_text})
    
    print(f"\n{'='*50}")
    print("EXPERIMENT COMPLETE!")
    print(f"Best overall performance: {best_overall} layers (RMSE: {all_results[best_overall]['overall_rmse']:.4f})")
    print(f"Best emergence prediction: {best_emergence} layers (RMSE: {all_results[best_emergence]['emergence_rmse']:.4f})")
    print(f"Best timing accuracy: {best_time_error} layers (Error: {all_results[best_time_error]['emergence_time_diff']:.2f} hrs)")
    print(f"{'='*50}")
    
    # Finish wandb run
    wandb.finish()

def lstm_ready(tile, size, power_maps, intensities, num_in, num_pred, model_seq_len=None):
    """Modified lstm_ready that adapts to available data length.
    
    If num_in is larger than available data, use the maximum possible sequence length.
    This ensures we can always generate some sequences for evaluation.
    
    Args:
        tile: Tile index
        size: Grid size
        power_maps: Power maps data
        intensities: Intensity data
        num_in: Requested input sequence length (AR-specific)
        num_pred: Number of prediction steps
        model_seq_len: Expected sequence length by the model (for padding)
    """
    # Get data for specific tile
    X_trans = power_maps[tile]  # (features, time)
    y_trans = intensities[tile]  # (time,)
    
    # Transpose to get time as first dimension
    X_trans = X_trans.T  # (time, features)
    
    available_time_steps = len(X_trans)
    
    # Calculate maximum possible num_in given the data and required num_pred
    max_possible_num_in = available_time_steps - num_pred
    
    if max_possible_num_in <= 0:
        raise ValueError(f"Not enough data for tile {tile}. Available: {available_time_steps}, Need at least: {num_pred + 1}")
    
    # Use the smaller of requested num_in or what's available
    effective_num_in = min(num_in, max_possible_num_in)
    
    # Split into sequences using effective sequence length
    X_ss, y_mm = split_sequences(X_trans, y_trans, effective_num_in, num_pred)
    
    # If model expects a different sequence length, pad accordingly
    target_seq_len = model_seq_len if model_seq_len is not None else effective_num_in
    if effective_num_in < target_seq_len and len(X_ss) > 0:
        padding_length = target_seq_len - effective_num_in
        # Pad with zeros at the beginning (older time steps)
        padding_shape = (len(X_ss), padding_length, X_ss.shape[2])
        padding = np.zeros(padding_shape)
        X_ss = np.concatenate([padding, X_ss], axis=1)
    
    # Convert to tensors
    X = torch.Tensor(X_ss)  # (batch, seq_len, input_dim)
    y = torch.Tensor(y_mm)  # (batch, output_dim)
    
    return X, y

if __name__ == "__main__":
    main()