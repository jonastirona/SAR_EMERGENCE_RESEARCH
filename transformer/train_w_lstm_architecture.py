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

# Import the evaluation module
from transformer.eval import evaluate_models_for_ar

# Set up logging
logging.basicConfig(
    level=logging.WARNING,  # Changed from INFO to WARNING
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

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

# Global variable to store normalization statistics
GLOBAL_NORM_STATS = None

def run_single_experiment(config: Dict[str, Any], device: torch.device, learning_rate: float, global_step_offset: int = 0) -> Dict[str, Any]:
    """Run a single experiment with the given configuration, tile-by-tile, fresh optimizer per tile (LSTM style)."""
    print(f"Training model with learning rate {learning_rate} (tile-by-tile, fresh optimizer per tile)...")

    # ARs list (copied from train_w_stats.py)
    ARs = [11130,11149,11158,11162,11199,11327,11344,11387,11393,11416,11422,11455,11619,11640,11660,11678,11682,11765,11768,11776,11916,11928,12036,12051,12085,12089,12144,12175,12203,12257,12331,12494,12659,12778,12864,12877,12900,12929,13004,13085,13098]
    size = 9
    rid_of_top = config['rid_of_top']
    num_in = config['num_in']
    num_pred = config['num_pred']
    n_epochs = config['n_epochs']
    batch_size = 128

    all_inputs, all_intensities = load_all_ars_data(ARs, rid_of_top, size, num_in, num_pred)
    input_size = np.shape(all_inputs)[1]
    remaining_rows = size - 2*rid_of_top
    tiles = remaining_rows * size

    print(f"\n=== DATA SHAPES ===")
    print(f"all_inputs shape: {all_inputs.shape}")
    print(f"all_intensities shape: {all_intensities.shape}")
    print(f"input_size (features): {input_size}")
    print(f"Tiles after trimming: {tiles}")

    # Model definition (shared, but re-initialized for each AR if desired)
    def make_model():
        return SpatioTemporalTransformer(
            input_dim=5,
            seq_len=num_in,
            embed_dim=config['embed_dim'],
            num_heads=4,
            ff_dim=config['ff_dim'],
            num_layers=config['num_layers'],
            output_dim=num_pred,
            dropout=config['dropout'],
        ).to(device)

    # Store results for all ARs/tiles
    all_training_results = {}
    # lr_curves = {}    # Store learning rate curves for each tile

    # Create a single model instance (like LSTM)
    model = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=n_epochs//10, gamma=0.9)
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
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=n_epochs//10, gamma=0.9)
            
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                
                current_lr = scheduler.get_last_lr()[0]
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
    models_dir = os.path.join(project_root, 'transformer', 'results', 'tile_models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"transformer_lstm_style_final_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Saved final model to {model_path}")
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
                'embed_dim': config['embed_dim'],
                'num_heads': 4,
                'ff_dim': config['ff_dim'],
                'num_layers': config['num_layers'],
                'dropout': config['dropout'],
                'rid_of_top': config['rid_of_top'],
                'num_pred': config['num_pred'],
                'time_window': config['time_window'],
                'num_in': config['num_in'],
                'hidden_size': config['embed_dim'],
                'learning_rate': learning_rate
            }
            temp_output_dir = f"/tmp/ar_eval_AR{ar}"
            os.makedirs(temp_output_dir, exist_ok=True)
            plot_path = evaluate_models_for_ar(ar, lstm_path, temp_model_path, transformer_params, temp_output_dir)
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
    summary_stats = {
        'total_tiles_trained': len(all_training_results),
        'successful_ar_evaluations': len(successful_ars),
        'failed_ar_evaluations': len(failed_ars),
        'avg_final_test_loss': np.mean([r['final_test_loss'] for r in all_training_results.values() if not np.isnan(r['final_test_loss'])]),
        'avg_emergence_rmse': np.mean([r['emergence_rmse'] for r in all_training_results.values() if not np.isnan(r['emergence_rmse'])]),
        'avg_overall_rmse': np.mean([r['overall_rmse'] for r in all_training_results.values() if not np.isnan(r['overall_rmse'])])
    }
    wandb.log(summary_stats)
    return all_training_results

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
    
    plt.tight_layout(rect=(0, 0.05, 1, 0.93))  # Better spacing
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
    
    plt.tight_layout(rect=(0, 0.05, 1, 0.93))  # Better spacing
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
    
    plt.tight_layout(rect=(0, 0.1, 0.85, 0.92))
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
    
    plt.tight_layout(rect=(0, 0.05, 1, 0.93))  # Better spacing
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
    
    plt.tight_layout(rect=(0, 0.05, 1, 0.93))
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
    
    plt.tight_layout(rect=(0.25, 0.15, 1, 0.93))
    
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
    print("Starting tile-by-tile transformer training (LSTM architecture style)...")
    
    # Read API key from .env file
    with open('/mmfs1/project/mx6/jst26/SAR_EMERGENCE_RESEARCH/.env', 'r') as f:
        for line in f:
            if line.startswith('WANDB_API_KEY='):
                api_key = line.strip().split('=')[1]
                break
    
    # Login to wandb first
    wandb.login(key=api_key)
    
    # Fixed configuration for the experiment
    config = {
        'num_pred': 12,
        'num_in': 110,
        'embed_dim': 64,
        'ff_dim': 128,
        'num_layers': 3,  # Fixed to 3 layers
        'dropout': 0.1,
        'n_epochs': 1000,  # UPDATED from 300 to 1000
        'time_window': 12,
        'rid_of_top': 4
    }
    
    # Create models directory for saving best models
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    results_dir = os.path.join(project_root, 'transformer', 'results')
    search_type = "tile_by_tile_training"  # This is a tile-by-tile training experiment
    models_dir = os.path.join(results_dir, search_type)
    os.makedirs(models_dir, exist_ok=True)
    print(f"Models will be saved to: {models_dir}")
    
    # Initialize wandb run
    wandb.init(
        project="sar-emergence",
        entity="jonastirona-new-jersey-institute-of-technology",
        config=config,
        name="transformer w lstm architecture and params",
        notes="Tile-by-tile training: t12 r4 i110 n3 h64 e1000 l0.01, 4 heads, 0.1 dropout. Transformer vs LSTM benchmark.",
    )
    
    # Run single experiment with tile-by-tile training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    learning_rate = 0.01  # Fixed learning rate
    
    print(f"\n{'='*50}")
    print(f"TILE-BY-TILE TRAINING: Learning rate {learning_rate}")
    print(f"{'='*50}")
    
    try:
        # Run tile-by-tile experiment
        all_results = run_single_experiment(config, device, learning_rate, 0)
        print(f"✓ Tile-by-tile training completed successfully")
        
    except Exception as e:
        print(f"✗ ERROR in tile-by-tile training:")
        print(f"  Exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\n{'='*50}")
    print("EXPERIMENT COMPLETE!")
    print(f"Tile-by-tile training finished with learning rate {learning_rate}")
    print(f"Total tiles trained: {len(all_results)}")
    print(f"{'='*50}")
    
    # Print model saving summary
    print(f"\n{'='*50}")
    print("SAVED MODELS SUMMARY")
    print(f"{'='*50}")
    print(f"Local directory: {models_dir}")
    print(f"Search type: {search_type}")
    print(f"Wandb artifacts: Individual tile models + evaluation plots")
    
    # Count saved models
    saved_models = 0
    for tile_key in all_results.keys():
        model_path = os.path.join(models_dir, f"{tile_key}_best_model.pth")
        if os.path.exists(model_path):
            saved_models += 1
    
    print(f"Saved models: {saved_models}/{len(all_results)} tiles")
    print(f"{'='*50}")
    
    # Finish wandb run
    wandb.finish()

def lstm_ready(tile, size, power_maps, intensities, num_in, num_pred):
    """LSTM-style data preprocessing for transformer compatibility.
    
    Uses the same data preprocessing approach as the LSTM code:
    - Transposes power_maps to (time, features, tiles)
    - Transposes intensities to (time, tiles)
    - Extracts specific tile data
    - Splits into sequences
    
    Args:
        tile: Tile index
        size: Grid size
        power_maps: Power maps data
        intensities: Intensity data
        num_in: Requested input sequence length
        num_pred: Number of prediction steps
    """
    # LSTM-style data preprocessing
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