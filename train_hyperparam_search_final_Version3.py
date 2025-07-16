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
from dataclasses import dataclass, asdict
from sklearn.metrics import r2_score
import itertools
import time
import pickle
import json

# Add project root to Python path 
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Import from existing files
from transformer.models.st_transformer import SpatioTemporalTransformer
from transformer.functions import smooth_with_numpy, emergence_indication, split_sequences, lstm_ready

# Import the evaluation module 
from transformer.eval import evaluate_models_for_ar

# Set up logging
logging.basicConfig(
    level=logging.WARNING,  # Changed from INFO to WARNING
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

@dataclass
class HyperparameterConfig:
    """Configuration class for hyperparameters - ONLY NEW ADDITION"""
    # Architecture parameters
    embed_dim: int = 64
    num_heads: int = 4
    ff_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.1
    
    # Training parameters
    learning_rate: float = 0.01
    weight_decay: float = 0.0
    optimizer_type: str = 'adam'
    scheduler_type: str = 'step_lr'
    batch_size: int = None  # None means no batching (original approach)
    gradient_clip_norm: float = 1.0
    
    # Fixed parameters
    num_pred: int = 12
    num_in: int = 110
    rid_of_top: int = 4
    n_epochs: int = 1000  # SAME AS ORIGINAL
    time_window: int = 12

def create_scheduler(optimizer, scheduler_type, n_epochs):
    """Create different types of learning rate schedulers - EXTENSION OF ORIGINAL"""
    if scheduler_type == 'step_lr':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=n_epochs//10, gamma=0.9)
    elif scheduler_type == 'reduce_lr':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=50, verbose=False
        )
    elif scheduler_type == 'warmup_cosine':
        return get_warmup_cosine_scheduler(optimizer, n_epochs)
    else:  # constant
        return None

def get_warmup_cosine_scheduler(optimizer, n_epochs):
    """Warmup + Cosine Annealing scheduler - NEW OPTION"""
    warmup_epochs = n_epochs // 10
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch) / float(max(1, warmup_epochs))
        else:
            progress = (epoch - warmup_epochs) / (n_epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate R² score."""
    if len(y_true.shape) > 1:
        y_true = y_true.flatten()
    if len(y_pred.shape) > 1:
        y_pred = y_pred.flatten()
    result = r2_score(y_true, y_pred)
    return float(result)

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
    
    # Process each AR individually with per-AR normalization (like LSTM)
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
        
        # Stack inputs and normalize PER AR (like LSTM)
        stacked_maps = np.stack([power_maps23, power_maps34, power_maps45, power_maps56], axis=1); stacked_maps[np.isnan(stacked_maps)] = 0
        
        # Per-AR normalization (Local normalization) - same as LSTM
        min_p = np.min(stacked_maps); max_p = np.max(stacked_maps)
        min_m = np.min(mag_flux); max_m = np.max(mag_flux)
        min_i = np.min(intensities); max_i = np.max(intensities)
        
        # Apply per-AR normalization
        stacked_maps = (stacked_maps - min_p) / (max_p - min_p)
        mag_flux = (mag_flux - min_m) / (max_m - min_m)
        intensities = (intensities - min_i) / (max_i - min_i)
        
        # Reshape mag_flux to have an extra dimension and then put it with pmaps
        mag_flux_reshaped = np.expand_dims(mag_flux, axis=1)
        pm_and_flux = np.concatenate([stacked_maps, mag_flux_reshaped], axis=1)
        
        # append all ARs
        all_inputs.append(pm_and_flux)
        all_intensities.append(intensities)
    
    all_inputs = np.stack(all_inputs, axis=-1)
    all_intensities = np.stack(all_intensities, axis=-1)
    
    print(f"Per-AR normalization applied (like LSTM)")
    print(f"  Each AR normalized to [0,1] using its own min/max values")
    print(f"  Preserves relative distances within each AR")
    print(f"all_inputs shape: {all_inputs.shape}")
    print(f"all_intensities shape: {all_intensities.shape}")
    
    return all_inputs, all_intensities

# ========== CORE TRAINING FUNCTION (EXACT COPY) ==========

def run_single_configuration(config: HyperparameterConfig, device: torch.device, experiment_idx: int, scheduler_name = None) -> Dict[str, Any]:
    
    print(f"Training model with config {experiment_idx} (tile-by-tile, fresh optimizer per tile)...")
    
    ARs = [11130,11149,11158,11162,11199,11327,11344,11387,11393,11416,11422,11455,11619,11640,11660,11678,11682,11765,11768,11776,11916,11928,12036,12051,12085,12089,12144,12175,12203,12257,12331,12494,12659,12778,12864,12877,12900,12929,13004,13085,13098]
    size = 9
    rid_of_top = config.rid_of_top
    num_in = config.num_in
    num_pred = config.num_pred
    n_epochs = config.n_epochs
    batch_size = 128
    
    if scheduler_name is None:
        scheduler_name = config.scheduler_type

    all_inputs, all_intensities = load_all_ars_data(ARs, rid_of_top, size, num_in, num_pred)
    input_size = np.shape(all_inputs)[1]
    remaining_rows = size - 2*rid_of_top
    tiles = remaining_rows * size
    
    print(f"\n=== DATA SHAPES ===")
    print(f"all_inputs shape: {all_inputs.shape}")
    print(f"all_intensities shape: {all_intensities.shape}")
    print(f"input_size (features): {input_size}")
    print(f"Tiles after trimming: {tiles}")

    def make_model():
        return SpatioTemporalTransformer(
            input_dim=5,
            seq_len=num_in,
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            ff_dim=config.ff_dim,
            num_layers=config.num_layers,
            output_dim=num_pred,
            dropout=config.dropout,
            use_cls_token=True,        # Enable for new training
            use_attention_pool=True,   # Enable for new training
            use_pre_mlp_norm=True      # Enable MLP norm as you requested
        ).to(device)

    # Store results for all ARs/tiles 
    all_training_results = {}

    # Create a single model instance (like LSTM)
    model = make_model()
    loss_fn = nn.MSELoss()

    for ar_idx, AR in enumerate(ARs):
        power_maps = all_inputs[:,:,:,ar_idx]
        intensities = all_intensities[:,:,ar_idx]
        print(f"\nAR {AR} - power_maps shape: {power_maps.shape}, intensities shape: {intensities.shape}")
        
        for tile in range(tiles):
            print(f"  Training AR {AR} - Tile {tile}")
            # Prepare data for this tile
            X_tile, y_tile = lstm_ready(tile, size, power_maps, intensities, num_in, num_pred)
            if X_tile.shape[0] == 0:
                print(f"    Skipping tile {tile} (no data)")
                continue
            X_tile = torch.reshape(X_tile, (X_tile.shape[0], num_in, X_tile.shape[2])).to(device)
            y_tile = y_tile.to(device)
            # Split train/test (80/20)
            train_size = int(0.8 * len(X_tile))
            X_train, X_test = X_tile[:train_size], X_tile[train_size:]
            y_train, y_test = y_tile[:train_size], y_tile[train_size:]
            
            # Fresh optimizer for each tile (like LSTM) 
            if config.optimizer_type == 'adamw':
                optimizer = torch.optim.AdamW(
                    model.parameters(), 
                    lr=config.learning_rate, 
                    weight_decay=config.weight_decay
                )
            else:  # adam 
                optimizer = torch.optim.Adam(
                    model.parameters(), 
                    lr=config.learning_rate, 
                    weight_decay=config.weight_decay
                )
            
            # Scheduler
            scheduler = create_scheduler(optimizer, config.scheduler_type, n_epochs)
            
            train_losses = []
            test_losses = []
            lr_values = []
            best_test_loss = float('inf')
            best_model_state = None
                
            for epoch in range(n_epochs):
                model.train()
                optimizer.zero_grad()
                outputs = model(X_train)
                loss = loss_fn(outputs, y_train)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.gradient_clip_norm)
                optimizer.step()
                
                # Update scheduler
                if scheduler is not None:
                    if config.scheduler_type == 'reduce_lr':
                        # ReduceLROnPlateau needs validation loss
                        model.eval()
                        with torch.no_grad():
                            test_pred = model(X_test)
                            test_loss_for_scheduler = loss_fn(test_pred, y_test)
                        scheduler.step(test_loss_for_scheduler.item())
                        model.train()
                    else:
                        scheduler.step()
                
                current_lr = optimizer.param_groups[0]['lr'] if scheduler else config.learning_rate
                lr_values.append(current_lr)
                train_losses.append(loss.item())
                
                # Test
                model.eval()
                with torch.no_grad():
                    test_pred = model(X_test)
                    test_loss = loss_fn(test_pred, y_test)
                    test_losses.append(test_loss.item())
                    if test_loss.item() < best_test_loss:
                        best_test_loss = test_loss.item()
                        best_model_state = model.state_dict().copy()
                        
                if epoch % max(1, n_epochs//10) == 0:
                    print(f"    Epoch {epoch}: train loss {loss.item():.5f}, test loss {test_loss.item():.5f}, lr {current_lr:.2e}")
                    
                wandb.log({
                    f"AR{AR}_Tile{tile}/train_loss": loss.item(),
                    f"AR{AR}_Tile{tile}/test_loss": test_loss.item(),
                    f"AR{AR}_Tile{tile}/learning_rate": current_lr,
                    "epoch": epoch
                })
                
            # Calculate final metrics 
            model.eval()
            with torch.no_grad():
                final_test_pred = model(X_test)
                final_test_loss = loss_fn(final_test_pred, y_test).item()
                
            y_test_np = y_test.cpu().numpy()
            final_test_pred_np = final_test_pred.cpu().numpy()
            emergence_metrics = calculate_emergence_metrics(y_test_np.flatten(), final_test_pred_np.flatten(), time_step=1.0)
            overall_metrics = calculate_tile_level_emergence_metrics(y_test_np, final_test_pred_np, np.zeros(len(y_test_np)))
            
            tile_key = f'AR{AR}_Tile{tile}'
            all_training_results[tile_key] = {
                'train_losses': train_losses,
                'test_losses': test_losses,
                'final_train_loss': train_losses[-1] if train_losses else float('nan'),
                'final_test_loss': final_test_loss,
                'emergence_rmse': emergence_metrics.get('emergence_rmse', float('nan')),
                'emergence_mae': emergence_metrics.get('emergence_mae', float('nan')),
                'emergence_mse': emergence_metrics.get('emergence_mse', float('nan')),
                'emergence_r2': emergence_metrics.get('emergence_r2', float('nan')),
                'overall_rmse': overall_metrics.get('overall_rmse', float('nan')),
                'overall_mae': overall_metrics.get('overall_mae', float('nan')),
                'overall_mse': overall_metrics.get('overall_mse', float('nan')),
                'overall_r2': overall_metrics.get('overall_r2', float('nan')),
            }
            
            wandb.log({
                f"AR{AR}_Tile{tile}/final_train_loss": train_losses[-1] if train_losses else float('nan'),
                f"AR{AR}_Tile{tile}/final_test_loss": final_test_loss,
                f"AR{AR}_Tile{tile}/emergence_rmse": emergence_metrics.get('emergence_rmse', float('nan')),
                f"AR{AR}_Tile{tile}/emergence_mae": emergence_metrics.get('emergence_mae', float('nan')),
                f"AR{AR}_Tile{tile}/emergence_mse": emergence_metrics.get('emergence_mse', float('nan')),
                f"AR{AR}_Tile{tile}/emergence_r2": emergence_metrics.get('emergence_r2', float('nan')),
                f"AR{AR}_Tile{tile}/overall_rmse": overall_metrics.get('overall_rmse', float('nan')),
                f"AR{AR}_Tile{tile}/overall_mae": overall_metrics.get('overall_mae', float('nan')),
                f"AR{AR}_Tile{tile}/overall_mse": overall_metrics.get('overall_mse', float('nan')),
                f"AR{AR}_Tile{tile}/overall_r2": overall_metrics.get('overall_r2', float('nan'))
            })
    
    print("\nTile-by-tile training complete. Results stored in all_training_results.")
    
    # Save the final model weights (like LSTM)
    models_dir = f'/mmfs1/project/mx6/sp3463/SAR_EMERGENCE_RESEARCH-main/transformer/results/{scheduler_name}_models'
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"transformer_hyperparam_exp_{experiment_idx}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")
    # Upload model to wandb
    wandb.save(model_path)
    
    # Create AR evaluation plots using the final model 
    print("\nCreating AR evaluation plots...")
    # LSTM model path (assuming it exists)
    lstm_path = "/mmfs1/project/mx6/jst26/SAR_EMERGENCE_RESEARCH/lstm/results/t12_r4_i110_n3_h64_e1000_l0.01.pth"
    # Test ARs to evaluate
    test_ars = [11698, 11726, 13165, 13179, 13183]
    successful_ars = []
    failed_ars = []
    
    for ar in test_ars:
        try:
            # Use the final model for evaluation (optionally, you can reload if needed)
            temp_model_path = model_path
            transformer_params = {
                'embed_dim': config.embed_dim,
                'num_heads': config.num_heads,
                'ff_dim': config.ff_dim,
                'num_layers': config.num_layers,
                'dropout': config.dropout,
                'rid_of_top': config.rid_of_top,
                'num_pred': config.num_pred,
                'time_window': config.time_window,
                'num_in': config.num_in,
                'hidden_size': config.embed_dim,
                'learning_rate': config.learning_rate
            }
            temp_output_dir = f'/mmfs1/project/mx6/sp3463/SAR_EMERGENCE_RESEARCH-main/transformer/results/{scheduler_name}_evaluation_plots/experiment_{experiment_idx:04d}'
            os.makedirs(temp_output_dir, exist_ok=True)
            plot_path = evaluate_models_for_ar(ar, lstm_path, temp_model_path, transformer_params, temp_output_dir)
            config_json_path = os.path.join(temp_output_dir, 'experiment_config.json')
            config_data = {
                'experiment_index': experiment_idx,
                'scheduler_type': scheduler_name,
                'model_path': model_path,
                'timestamp': datetime.now().isoformat(),
                'hyperparameters': asdict(config),
                'training_results': {
                    'total_tiles_trained': len(all_training_results),
                    'avg_final_test_loss': float(np.mean([r['final_test_loss'] for r in all_training_results.values() if not np.isnan(r['final_test_loss'])])),
                    'avg_emergence_rmse': float(np.mean([r['emergence_rmse'] for r in all_training_results.values() if not np.isnan(r['emergence_rmse'])])),
                    'avg_overall_rmse': float(np.mean([r['overall_rmse'] for r in all_training_results.values() if not np.isnan(r['overall_rmse'])]))
                },
                'evaluation_results': {
                    'successful_ars': successful_ars,
                    'failed_ars': failed_ars,
                    'success_rate': len(successful_ars) / len(test_ars) if test_ars else 0
                }
            }
            
            with open(config_json_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            print(f"Saved experiment config JSON to: {config_json_path}")
            if plot_path and os.path.exists(plot_path):
                wandb.log({f'AR_{ar}_comparison': wandb.Image(plot_path)})
                successful_ars.append(ar)
                print(f"  ✓ AR {ar} evaluation completed")
            else:
                failed_ars.append(ar)
                print(f"  ✗ AR {ar} evaluation failed")
        except Exception as e:
            failed_ars.append(ar)
            print(f"  ✗ Error evaluating AR {ar}: {str(e)}")
            continue
    
    print(f"AR evaluations completed: {len(successful_ars)}/{len(test_ars)} successful")
    

    
    # Summary stats 
    summary_stats = {
        'total_tiles_trained': len(all_training_results),
        'successful_ar_evaluations': len(successful_ars),
        'failed_ar_evaluations': len(failed_ars),
        'avg_final_test_loss': np.mean([r['final_test_loss'] for r in all_training_results.values() if not np.isnan(r['final_test_loss'])]),
        'avg_emergence_rmse': np.mean([r['emergence_rmse'] for r in all_training_results.values() if not np.isnan(r['emergence_rmse'])]),
        'avg_overall_rmse': np.mean([r['overall_rmse'] for r in all_training_results.values() if not np.isnan(r['overall_rmse'])])
    }
    wandb.log(summary_stats)
    
    return summary_stats

# ========== HYPERPARAMETER GRID SEARCH FUNCTIONS ==========

def generate_hyperparameter_combinations():
    """Generate all hyperparameter combinations for grid search."""
    
    # Define search space EXACTLY AS SPECIFIED
    search_space = {
        'learning_rate': [0.01], #0.005, 0.001
        'weight_decay': [0.0, 1e-4], #1e-5, 1e-4, 1e-3
        'optimizer_type': ['adam', 'adamw'],
        'num_layers': [3], #try 5?
        'num_heads': [4], #trys 8
        'embed_dim': [64], #128
        'ff_dim': [128], #256
        'dropout': [0.0, 0.1],
        'scheduler_type': ['reduce_lr'], #step_lr, 'warmup_cosine', reduce_lr
        'batch_size': [None, 64],  # None means no batching
        'gradient_clip_norm': [1.0] #0.5
    }
    
    # Generate all combinationss
    keys = list(search_space.keys())
    values = list(search_space.values())
    
    combinations = []
    for combination in itertools.product(*values):
        config_dict = dict(zip(keys, combination))
        
        # Validate that embed_dim is divisible by num_heads
        if config_dict['embed_dim'] % config_dict['num_heads'] != 0:
            continue
            
        combinations.append(config_dict)
    
    print(f"Generated {len(combinations)} valid hyperparameter combinations")
    return combinations

def evaluate_configuration_performance(results: Dict[str, float]) -> float:
    """Evaluate configuration performance with combined score (lower is better)."""
    emergence_rmse = results.get('avg_emergence_rmse', float('inf'))
    overall_rmse = results.get('avg_overall_rmse', float('inf'))
    emergence_r2 = results.get('avg_emergence_r2', 0.0)
    
    # Combined score (lower is better)
    # Prioritize emergence performance but include overall performance
    score = (0.7 * emergence_rmse + 0.3 * overall_rmse) - (0.1 * emergence_r2)
    return score

def create_experiment_name(config: HyperparameterConfig, idx: int) -> str:
    """Create descriptive experiment name."""
    return f"grid_exp_{idx:04d}_l{config.num_layers}h{config.num_heads}_e{config.embed_dim}_lr{config.learning_rate}_wd{config.weight_decay}_{config.scheduler_type}_{config.optimizer_type}"

def run_hyperparameter_grid_search(device: torch.device, max_experiments: int = None) -> Tuple[HyperparameterConfig, Dict]:
    """Run comprehensive hyperparameter grid search."""
    
    # Generate all parameter combinations
    param_combinations = generate_hyperparameter_combinations()
    
    if max_experiments is not None:
        param_combinations = param_combinations[:max_experiments]
        print(f"Limited to first {max_experiments} experiments")
    
    print(f"Running grid search with {len(param_combinations)} configurations")
    
    best_config = None
    best_score = float('inf')
    all_results = {}
    
    # Track progress
    start_time = time.time()
    
    for idx, params in enumerate(param_combinations):
        print(f"\n{'='*80}")
        print(f"GRID SEARCH EXPERIMENT {idx+1}/{len(param_combinations)}")
        print(f"{'='*80}")
        
        config = HyperparameterConfig(**params)
        
        # Create unique experiment name
        exp_name = create_experiment_name(config, idx)
        
        # Initialize WandB for this experiment
        wandb.init(
            project=f"sar-emergence-grid-search-{config.scheduler_type}",
            entity="jonastirona-new-jersey-institute-of-technology",
            config=asdict(config),
            name=exp_name,
            reinit=True
        )
        
        try:
            # Run single experiment with this configuration
            results = run_single_configuration(config, device, idx, config.scheduler_type)
            
            # Evaluate performance
            score = evaluate_configuration_performance(results)
            
            # Store results
            all_results[exp_name] = {
                'config': config,
                'results': results,
                'score': score,
                'experiment_idx': idx
            }
            
            # Track best configuration
            if score < best_score:
                best_score = score
                best_config = config
                print(f"🎯 NEW BEST CONFIGURATION! Score: {score:.6f}")
                print(f"   Config: {config}")
            
            # Log summary metrics to wandb
            wandb.log({
                'experiment_score': score,
                'is_best_so_far': score == best_score,
                'experiment_index': idx,
                **{f'final/{k}': v for k, v in results.items()}
            })
            
            # Print progress
            elapsed_time = time.time() - start_time
            avg_time_per_exp = elapsed_time / (idx + 1)
            estimated_total_time = avg_time_per_exp * len(param_combinations)
            remaining_time = estimated_total_time - elapsed_time
            
            print(f"✓ Experiment {idx+1} completed. Score: {score:.6f}")
            print(f"  Time: {elapsed_time/3600:.1f}h elapsed, {remaining_time/3600:.1f}h remaining")
            print(f"  Best score so far: {best_score:.6f}")
            
        except Exception as e:
            print(f"❌ Experiment {idx+1} failed: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
        finally:
            wandb.finish()
    
    return best_config, all_results

def save_results(best_config: HyperparameterConfig, all_results: Dict, output_dir: str):
    """Save hyperparameter search results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results as pickle
    results_path = os.path.join(output_dir, 'grid_search_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump({
            'best_config': best_config,
            'all_results': all_results,
            'timestamp': datetime.now().isoformat()
        }, f)
    
    # Save best config as text
    best_config_path = os.path.join(output_dir, 'best_config.txt')
    with open(best_config_path, 'w') as f:
        f.write("BEST HYPERPARAMETER CONFIGURATION\n")
        f.write("="*16 + "\n\n")
        for key, value in asdict(best_config).items():
            f.write(f"{key}: {value}\n")
    
    # Create summary CSV
    summary_data = []
    for exp_name, exp_data in all_results.items():
        config = exp_data['config']
        results = exp_data['results']
        row = {
            'experiment_name': exp_name,
            'score': exp_data['score'],
            **asdict(config),
            **results
        }
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('score')  # Sort by score (best first)
    summary_path = os.path.join(output_dir, 'grid_search_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    
    print(f"\nResults saved to:")
    print(f"  - Full results: {results_path}")
    print(f"  - Best config: {best_config_path}")
    print(f"  - Summary CSV: {summary_path}")

def main():
    """Main function for hyperparameter grid search."""
    print("Starting Comprehensive Hyperparameter Grid Search...")
    
    # Read API key
    try:
        with open('/mmfs1/project/mx6/jst26/SAR_EMERGENCE_RESEARCH/.env', 'r') as f:
            for line in f:
                if line.startswith('WANDB_API_KEY='):
                    api_key = line.strip().split('=')[1]
                    break
        wandb.login(key=api_key)
    except Exception as e:
        print(f"Warning: Could not set up wandb: {e}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Run hyperparameter search
    max_experiments = 16 # Adjust as needed
    best_config, all_results = run_hyperparameter_grid_search(
        device=device,
        max_experiments=max_experiments
    )
    
    print(f"\n{'='*80}")
    print("HYPERPARAMETER GRID SEARCH COMPLETE!")
    print(f"{'='*80}")
    print(f"Best Configuration:")
    for key, value in asdict(best_config).items():
        print(f"  {key}: {value}")
    
    # Save results
    results_dir = '/mmfs1/project/mx6/sp3463/SAR_EMERGENCE_RESEARCH-main/transformer/results/grid_search'
    save_results(best_config, all_results, results_dir)
    
    print(f"\nGrid search completed!")
    print(f"Total configurations tested: {len(all_results)}")
    if all_results:
        print(f"Best score: {min(r['score'] for r in all_results.values()):.6f}")

if __name__ == "__main__":
    main()