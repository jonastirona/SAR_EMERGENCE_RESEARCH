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

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Import from existing files
from transformer.models.st_transformer import SpatioTemporalTransformer
from transformer.functions import lstm_ready, smooth_with_numpy, emergence_indication, split_sequences

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
    print(f"Calculating tile-level emergence metrics for {len(np.unique(tile_indices))} unique tiles...")
    
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
            print(f"  Skipping tile {tile_idx}: insufficient data ({len(tile_obs_flat)} points)")
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
            print(f"  Error calculating metrics for tile {tile_idx}: {str(e)}")
            continue
    
    tile_metrics['num_valid_tiles'] = valid_tiles
    print(f"  Successfully calculated metrics for {valid_tiles}/{len(unique_tiles)} tiles")
    
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

def get_ar_settings(test_AR, rid_of_top):
    """Get AR-specific settings."""
    if test_AR == 11698:
        starting_tile = 46 - rid_of_top * 9
        before_plot = 50
        num_in = 96
        NOAA_first = datetime(2013, 3, 15)
        NOAA_second = datetime(2013, 3, 17)
    elif test_AR == 11726:
        starting_tile = 37 - rid_of_top * 9
        before_plot = 50
        num_in = 72
        NOAA_first = datetime(2013, 4, 20)
        NOAA_second = datetime(2013, 4, 22)
    elif test_AR == 13165:
        rid_of_top = 1
        starting_tile = 28 - rid_of_top * 9
        before_plot = 40
        num_in = 96
        NOAA_first = datetime(2022, 12, 12)
        NOAA_second = datetime(2022, 12, 14)
    elif test_AR == 13179:
        starting_tile = 37 - rid_of_top * 9
        before_plot = 40
        num_in = 96
        NOAA_first = datetime(2022, 12, 30)
        NOAA_second = datetime(2023, 1, 1)
    elif test_AR == 13183:
        starting_tile = 37 - rid_of_top * 9
        before_plot = 40
        num_in = 96
        NOAA_first = datetime(2023, 1, 6)
        NOAA_second = datetime(2023, 1, 8)
    else:
        raise ValueError("Invalid test_AR value")
    return starting_tile, before_plot, num_in, NOAA_first, NOAA_second

def recalibrate(pred, last_obs):
    """Recalibrate predictions to match the last observation."""
    trend = pred - pred[0]
    new_pred = trend + last_obs
    return new_pred

def evaluate_model_on_ar(model, test_AR, config, device):
    """Evaluate a trained model on a specific AR and return predictions and metrics."""
    size = 9
    rid_of_top = config['rid_of_top']
    
    # Get AR settings
    start_tile, before_plot, ar_num_in, NOAA_first, NOAA_second = get_ar_settings(test_AR, rid_of_top)
    
    # Update model's sequence length dynamically - like train_and_eval_transformer.py
    model.seq_len = ar_num_in
    model.positional_encoding = model._generate_positional_encoding(ar_num_in, config['embed_dim'])
    
    # Load AR data
    base_path = f'/mmfs1/project/mx6/jst26/SAR_EMERGENCE_RESEARCH/data/AR{test_AR}'
    power = np.load(f'{base_path}/mean_pmdop{test_AR}_flat.npz', allow_pickle=True)
    mag = np.load(f'{base_path}/mean_mag{test_AR}_flat.npz', allow_pickle=True)
    cont = np.load(f'{base_path}/mean_int{test_AR}_flat.npz', allow_pickle=True)
    
    pm23, pm34, pm45, pm56, time_arr = (
        power['arr_0'], power['arr_1'], power['arr_2'], power['arr_3'], power['arr_4']
    )
    mf = mag['arr_0']
    ii = cont['arr_0']
    
    # Trim arrays
    sl = slice(rid_of_top * size, -rid_of_top * size)
    pm23, pm34, pm45, pm56 = pm23[sl,:], pm34[sl,:], pm45[sl,:], pm56[sl,:]
    mf = mf[sl,:]
    ii = ii[sl,:]
    mf[np.isnan(mf)] = 0
    ii[np.isnan(ii)] = 0
    
    # Normalize using local AR statistics - exactly like eval_comparison.py
    # This creates more realistic, noisy data compared to global normalization
    stacked = np.stack([pm23, pm34, pm45, pm56], axis=1)
    mp, Mp = stacked.min(), stacked.max()
    mm, Mm = mf.min(), mf.max()
    mi, Mi = ii.min(), ii.max()
    stacked = (stacked - mp) / (Mp - mp)
    mf = (mf - mm) / (Mm - mm)
    ii = (ii - mi) / (Mi - mi)
    
    # Create inputs - exactly like eval_comparison.py
    inputs = np.concatenate([stacked, np.expand_dims(mf, 1)], axis=1)
    
    # Use the AR-specific num_in
    num_pred = config['num_pred']
    
    # Evaluate on main tile
    tile_idx = start_tile
    X_test, y_test = lstm_ready_original(tile_idx, size, inputs, ii, ar_num_in, num_pred)
    X_test = X_test.to(device)
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)[:, -1].cpu().numpy()  # Get last prediction
    
    true_values = y_test[:, -1].numpy()  # Get last true value
    
    # Recalibrate predictions
    last_obs_idx = ii.shape[1] - true_values.shape[0] - 1
    last_obs = ii[tile_idx, last_obs_idx]
    predictions = recalibrate(predictions, last_obs)
    
    # Calculate full sequence metrics (no short/long split)
    full_mse = np.mean((true_values - predictions) ** 2)
    full_rmse = np.sqrt(full_mse)
    full_mae = np.mean(np.abs(true_values - predictions))
    full_r2 = 1 - np.sum((true_values - predictions) ** 2) / np.sum((true_values - np.mean(true_values)) ** 2)
    
    # Calculate emergence metrics
    emergence_metrics = calculate_emergence_metrics(true_values, predictions)
    
    return {
        'ar': test_AR,
        'true_values': true_values,
        'predictions': predictions,
        'full_metrics': (full_mse, full_rmse, full_mae, full_r2),
        'emergence_metrics': emergence_metrics,
        'time_array': time_arr[last_obs_idx+1:last_obs_idx+1+len(true_values)] if len(time_arr) > last_obs_idx+len(true_values) else None,
        'NOAA_dates': (NOAA_first, NOAA_second)
    }

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

def create_derivative_plots(model, test_AR, config, device, num_layers) -> plt.Figure:
    """Create comprehensive 7-tile evaluation plots for a specific AR using actual model predictions."""
    try:
        print(f"Creating detailed evaluation plots for AR {test_AR} with {num_layers} layers...")
        
        size = 9
        # Force rid_of_top = 1 for evaluation, just like eval_comparison.py
        rid_of_top = 1
        num_in = config['num_in']  # Use model's trained sequence length (110)
        num_pred = config['num_pred']
        
        # Get AR-specific settings (using forced rid_of_top=1)
        start_tile, before_plot, ar_num_in, NOAA_first, NOAA_second = get_ar_settings(test_AR, rid_of_top)
        NOAA1 = mdates.date2num(NOAA_first)
        NOAA2 = mdates.date2num(NOAA_second)
        
        # Update model's sequence length dynamically - like train_and_eval_transformer.py
        model.seq_len = ar_num_in
        model.positional_encoding = model._generate_positional_encoding(ar_num_in, config['embed_dim'])
        
        # Load AR data
        base_path = f'/mmfs1/project/mx6/jst26/SAR_EMERGENCE_RESEARCH/data/AR{test_AR}'
        power = np.load(f'{base_path}/mean_pmdop{test_AR}_flat.npz', allow_pickle=True)
        mag = np.load(f'{base_path}/mean_mag{test_AR}_flat.npz', allow_pickle=True)
        cont = np.load(f'{base_path}/mean_int{test_AR}_flat.npz', allow_pickle=True)
        
        pm23, pm34, pm45, pm56, time_arr = (
            power['arr_0'], power['arr_1'], power['arr_2'], power['arr_3'], power['arr_4']
        )
        mf = mag['arr_0']
        ii = cont['arr_0']
        
        # Trim arrays (using rid_of_top=1)
        sl = slice(rid_of_top * size, -rid_of_top * size)
        pm23, pm34, pm45, pm56 = pm23[sl,:], pm34[sl,:], pm45[sl,:], pm56[sl,:]
        mf = mf[sl,:]
        ii = ii[sl,:]
        mf[np.isnan(mf)] = 0
        ii[np.isnan(ii)] = 0
        
        # Normalize using local AR statistics - exactly like eval_comparison.py
        # This creates more realistic, noisy data compared to global normalization
        stacked = np.stack([pm23, pm34, pm45, pm56], axis=1)
        mp, Mp = stacked.min(), stacked.max()
        mm, Mm = mf.min(), mf.max()
        mi, Mi = ii.min(), ii.max()
        stacked = (stacked - mp) / (Mp - mp)
        mf = (mf - mm) / (Mm - mm)
        ii = (ii - mi) / (Mi - mi)
        
        # Create inputs - exactly like eval_comparison.py
        inputs = np.concatenate([stacked, np.expand_dims(mf, 1)], axis=1)
        
        # Create figure with exact same styling as eval_comparison.py
        fig = plt.figure(figsize=(16, 46))
        fig.subplots_adjust(left=0.15, right=0.85, top=0.97, bottom=0.1)
        gs0 = gridspec.GridSpec(7, 1, figure=fig, hspace=0.2)
        
        # Parameters from eval_comparison.py
        fut = num_pred - 1  # Get last prediction
        thr = -0.01  # Threshold for emergence detection
        st = 4  # Sustained time (4 hours)
        
        for i in range(7):
            tile_idx = start_tile + i
            disp = tile_idx + 10
            print(f"  Processing tile {disp}...")
            
            # Get model predictions for this tile - use AR-specific num_in like train_and_eval_transformer.py
            X_test, y_test = lstm_ready_original(tile_idx, size, inputs, ii, ar_num_in, num_pred)
            X_test = X_test.to(device)
            
            model.eval()
            with torch.no_grad():
                p_t = model(X_test)[:, fut].cpu().numpy()  # Get last prediction, same as eval_comparison
            
            true = y_test[:, fut].numpy()  # Get last true value, same as eval_comparison
            
            # Recalibrate predictions - same as eval_comparison
            last = ii.shape[1] - true.shape[0] - 1
            p_t = recalibrate(p_t, ii[tile_idx, last])
            
            # Calculate overall metrics
            overall_mse = np.mean((true - p_t) ** 2)
            overall_rmse = np.sqrt(overall_mse)
            overall_mae = np.mean(np.abs(true - p_t))
            overall_r2 = 1 - np.sum((true - p_t) ** 2) / np.sum((true - np.mean(true)) ** 2)
            
            # Calculate emergence window metrics
            emergence_metrics = calculate_emergence_metrics(true, p_t)
            
            # Calculate emergence timing difference for metrics table
            def find_emergence_start(data, threshold=-0.01, min_duration=4):
                """Find the first emergence start time in hours."""
                d_data = np.gradient(data)
                consecutive_negative = 0
                for i, val in enumerate(d_data):
                    if val < threshold:
                        consecutive_negative += 1
                        if consecutive_negative >= min_duration:
                            return i - min_duration + 1  # Return start of the 4-hour period
                    else:
                        consecutive_negative = 0
                return None
            
            observed_emergence_start = find_emergence_start(true)
            predicted_emergence_start = find_emergence_start(p_t)
            
            emergence_time_diff = None
            if observed_emergence_start is not None and predicted_emergence_start is not None:
                emergence_time_diff = abs(predicted_emergence_start - observed_emergence_start)
            
            # Prepare data for plotting - exact same as train_and_eval_transformer.py
            before = ii[tile_idx, last-before_plot:last]
            tcut = time_arr[last-before_plot:last+true.shape[0]]
            tnum = mdates.date2num(tcut)
            nanarr = np.full(before.shape, np.nan)
            
            # Calculate derivatives for the FULL sequence - like train_and_eval_transformer.py
            full_true = np.concatenate((before, true))
            d_obs = np.gradient(smooth_with_numpy(full_true))  # Full observed derivative
            d_pred = np.gradient(p_t)  # Just prediction derivative
            
            # Find emergence window based on actual dObs/dt for THIS tile
            ind_o = emergence_indication(d_obs, thr, st)
            emergence_detected = np.any(ind_o != 0)
            emergence_start_idx = None
            emergence_end_idx = None
            
            if emergence_detected:
                # Find the first and last emergence points in the full timeline
                emergence_indices = np.where(ind_o != 0)[0]
                if len(emergence_indices) > 0:
                    emergence_start_idx = emergence_indices[0]
                    emergence_end_idx = emergence_indices[-1]
            
            # Derivative plots: pad with NaNs for the "before_plot" segment - exact same as eval_comparison
            # Create NaN padding for the pre-prediction window
            nan_pad = np.full(before_plot, np.nan)
            # Full-length derivative arrays
            d_t_full = np.concatenate([nan_pad, d_pred])
            
            # Create subplots for this tile - exact same height ratios and spacing as eval_comparison
            gs1 = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs0[i], 
                                               height_ratios=[18, 4, 4, 4], hspace=0.3)
            
            # 1. Main intensity plot - exact same styling as eval_comparison
            ax0 = fig.add_subplot(gs1[0])
            ax0.plot(tnum, np.concatenate((before, true)), 'k-', label='Observed Intensity')
            ax0.plot(tnum, np.concatenate((nanarr, p_t)), 'r-', label='Transformer Prediction')
            ax0.axvline(NOAA1, color='magenta', linestyle='--', label='NOAA First Record')
            ax0.axvline(NOAA2, color='darkmagenta', linestyle='--', label='NOAA Second Record')
            
            # Add emergence window shading based on actual dObs/dt emergence for this tile
            if emergence_detected and emergence_start_idx is not None and emergence_end_idx is not None:
                # Convert indices to time values
                window_start_time = tnum[emergence_start_idx]
                window_end_time = tnum[min(emergence_end_idx, len(tnum) - 1)]
                ax0.axvspan(window_start_time, window_end_time, alpha=0.2, color='yellow', 
                           label='Emergence Window')
            
            ax0.set_title(f'Tile {disp}', fontsize=12)
            ax0.set_ylabel('Normalized Intensity', fontsize=9, labelpad=20)
            ax0.set_ylim([-0.1, 1.1])
            ax0.grid(True)
            ax0.set_yticks([0, 0.25, 0.5, 0.75, 1])
            # Hide x-axis labels for main plot - dates only on bottom
            ax0.tick_params(labelbottom=False)
            legend = ax0.legend(bbox_to_anchor=(1.033, .83, 0.223, 0.11), loc='upper left', 
                              borderaxespad=0, fontsize=10, framealpha=1, mode='expand')
            legend.get_frame().set_boxstyle('square', pad=1)
            
            # Add metrics table - overall and emergence metrics only
            def create_transformer_metrics_table(ax, overall_metrics, emergence_metrics, timing_diff):
                # Create table data with overall and emergence metrics only
                data = [
                    ['Metric', 'Value'],
                    ['Overall', ''],
                    ['MSE', f'{overall_metrics[0]:.4f}'],
                    ['RMSE', f'{overall_metrics[1]:.4f}'],
                    ['MAE', f'{overall_metrics[2]:.4f}'],
                    ['R²', f'{overall_metrics[3]:.4f}'],
                    ['', ''],
                    ['Emergence', ''],  # Shortened from "Emergence Window" to fit
                    ['RMSE', f'{emergence_metrics["emergence_rmse"]:.4f}'],
                    ['MAE', f'{emergence_metrics["emergence_mae"]:.4f}'],
                    ['R²', f'{emergence_metrics["emergence_r2"]:.4f}']
                ]
                
                # Add timing difference if available
                if timing_diff is not None:
                    data.append(['Time Diff', f'{timing_diff:.1f}h'])
                else:
                    data.append(['Time Diff', 'N/A'])
                
                # Create table - positioned lower like requested
                table = ax.table(
                    cellText=data,
                    loc='upper left',
                    bbox=[1.02, -0.5, 0.22, 0.75],  # Slightly taller for extra row
                    cellLoc='center',
                    colLoc='center'
                )
                
                # Style the table exactly like eval_comparison
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                
                # Style cells
                for (row, col), cell in table.get_celld().items():
                    cell.set_text_props(color='black')
                    cell.set_facecolor('white')
                    cell.set_edgecolor('#CCCCCC')
                    cell.set_linewidth(0.5)
            
            create_transformer_metrics_table(ax0, (overall_mse, overall_rmse, overall_mae, overall_r2), emergence_metrics, emergence_time_diff)
            
            # 2. Observed derivative with emergence detection - exact same as eval_comparison
            ax1 = fig.add_subplot(gs1[1], sharex=ax0)
            # dObs/dt: plot full black, overlay limegreen for emergence
            ax1.plot(tnum, d_obs, color='black', linewidth=1)
            for j in range(len(d_obs)-1):
                if ind_o[j] != 0:
                    ax1.plot(tnum[j:j+2], d_obs[j:j+2], color='limegreen', linewidth=1)
            ax1.set_ylabel('dObs/dt', fontsize=7, labelpad=10)
            ax1.set_ylim([-0.05, 0.05])
            ax1.set_yticks([0])
            ax1.grid(True)
            ax1.tick_params(labelbottom=False)  # Hide x-axis labels
            ax1.xaxis.set_major_locator(mdates.DayLocator())
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            
            # 3. Predicted derivative with emergence detection - exact same as eval_comparison
            ax2 = fig.add_subplot(gs1[2], sharex=ax0)
            # dTrans/dt: plot full red, overlay limegreen for emergence
            ax2.plot(tnum, d_t_full, color='red', linewidth=1)
            ind_t = emergence_indication(d_t_full, thr, st)
            for j in range(len(d_t_full)-1):
                if ind_t[j] != 0:
                    ax2.plot(tnum[j:j+2], d_t_full[j:j+2], color='limegreen', linewidth=1)
            ax2.set_ylabel('dTrans/dt', fontsize=7, labelpad=10)
            ax2.set_ylim([-0.05, 0.05])
            ax2.set_yticks([0])
            ax2.grid(True)
            ax2.tick_params(labelbottom=False)  # Hide x-axis labels
            ax2.xaxis.set_major_locator(mdates.DayLocator())
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax2.set_xlim(tnum[0], tnum[-1])
            
            # 4. Error curve - exact same as eval_comparison - ONLY subplot with dates shown
            ax3 = fig.add_subplot(gs1[3], sharex=ax0)
            ax3.plot(tnum[before_plot:before_plot+len(true)], np.abs(true - p_t), 'r-')
            ax3.axvline(NOAA1, color='magenta', linestyle='--')
            ax3.set_ylabel('|Error|', fontsize=8)
            ax3.set_xlabel('Date', fontsize=10)
            ax3.set_xlim(tnum[0], tnum[-1])
            ax3.grid(True)
            ax3.xaxis.set_major_locator(mdates.DayLocator())
            ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax3.tick_params(labelbottom=True)  # Show x-axis labels only here
        
        plt.tight_layout(rect=[0, 0, 0.8, 0.96])
        plt.subplots_adjust(right=0.8)
        plt.suptitle(f'Model Comparison for AR {test_AR}', y=0.99)
        return fig
        
    except Exception as e:
        print(f"Error creating detailed plots for AR {test_AR}: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Return a simple error plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.text(0.5, 0.5, f'Error evaluating AR {test_AR}:\n{str(e)}', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title(f'AR {test_AR} Evaluation Error')
        return fig

def run_single_experiment(config: Dict[str, Any], device: torch.device, num_layers: int) -> Dict[str, float]:
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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
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
            
            # Log per-epoch metrics including new ones with layer prefix
            wandb.log({
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
                f'layers_{num_layers}/learning_rate': optimizer.param_groups[0]['lr']
            }, step=epoch + (num_layers - 1) * config['n_epochs'])
            
            if not np.isnan(epoch_test_loss) and epoch_test_loss < best_test_loss:
                best_test_loss = epoch_test_loss
                best_predictions = all_test_preds
                best_observations = all_test_targets
                best_model_state = model.state_dict()
                best_tile_indices = tile_indices_test
        
        scheduler.step(epoch_test_loss)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1}/{config['n_epochs']} - Train Loss: {epoch_train_loss:.5f}, Test Loss: {epoch_test_loss:.5f}")
    
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
        'num_pred': 24,
        'num_in': 110,
        'embed_dim': 64,
        'num_heads': 8,
        'ff_dim': 128,
        'dropout': 0.1,
        'learning_rate': 0.001,
        'n_epochs': 10,
        'time_window': 12,
        'rid_of_top': 4
    }
    
    # Initialize wandb run
    wandb.init(
        project="sar-emergence",
        entity="jonastirona-new-jersey-institute-of-technology",
        config=config,
        name="SAR_Transformer_LayerSearch_1to5_110seq_24pred_EmergenceAnalysis",
        notes="Comprehensive grid search comparing transformer performance across 1-5 layers with AR emergence evaluation"
    )
    
    # Run experiments for 1-5 layers
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_results = {}
    
    layer_range = range(1, 6)  # 1 to 5 layers
    for num_layers in layer_range:
        print(f"\n{'='*50}")
        print(f"TRIAL {num_layers}/5: Testing {num_layers} layer(s)")
        print(f"{'='*50}")
        
        # Run experiment
        results = run_single_experiment(config, device, num_layers)
        all_results[num_layers] = results
        
        # Create individual plots for each AR using the best trained model
        test_ars = [11698, 11726, 13165, 13179, 13183]
        print(f"Generating plots for test ARs...")
        
        # Load the best model state for evaluation
        eval_model = SpatioTemporalTransformer(
            input_dim=5,
            seq_len=config['num_in'],
            embed_dim=config['embed_dim'],
            num_heads=config['num_heads'],
            num_layers=num_layers,
            ff_dim=config['ff_dim'],
            dropout=config['dropout'],
            output_dim=config['num_pred']
        ).to(device)
        
        if results['best_model_state'] is not None:
            eval_model.load_state_dict(results['best_model_state'])
        
        for ar in test_ars:
            print(f"  Evaluating AR {ar} with {num_layers} layers...")
            fig = create_derivative_plots(
                eval_model,
                ar,
                config,
                device,
                num_layers
            )
            wandb.log({f'layers_{num_layers}/ar_{ar}_plots': wandb.Image(fig)})
            plt.close(fig)
            print(f"  Completed AR {ar} evaluation.")
    
    # Create comprehensive comparison plots
    print(f"\n{'='*50}")
    print("Creating comprehensive comparison plots...")
    print(f"{'='*50}")
    create_cross_trial_comparison_plots(all_results, config)
    
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

def lstm_ready_original(tile, size, power_maps, intensities, num_in, num_pred):
    """Original lstm_ready function without padding - like train_and_eval_transformer.py."""
    # Transpose to match expected format
    final_maps = np.transpose(power_maps, axes=(2, 1, 0))
    final_ints = np.transpose(intensities, axes=(1, 0))
    X_trans = final_maps[:, :, tile]
    y_trans = final_ints[:, tile]
    X_ss, y_mm = split_sequences(X_trans, y_trans, num_in, num_pred)
    return torch.Tensor(X_ss), torch.Tensor(y_mm)

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
    
    print(f"  Tile {tile}: Using seq_len={effective_num_in} (requested={num_in}, available={available_time_steps})")
    
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