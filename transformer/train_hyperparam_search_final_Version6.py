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
from torch.nn.utils.rnn import pad_sequence

# Add project root to Python path 
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Import from existing files
from transformer.models.st_transformer_new import SpatioTemporalTransformer
from transformer.functions_conv import (
    smooth_with_numpy, emergence_indication, split_sequences, lstm_ready,
    get_neighbor_tiles
)

# Import the evaluation module 
from transformer.eval_full_sequence_v6 import evaluate_models_for_ar

# Set up logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

@dataclass
class HyperparameterConfig:
    """Configuration class for hyperparameters - Version 5 with Temporal Conv"""
    # Architecture parameters
    embed_dim: int = 128
    num_heads: int = 4
    ff_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.0
    
    # NEW: Temporal convolution parameters
    use_temporal_conv: bool = True  # Enable temporal convolution by default
    
    # Training parameters
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    optimizer_type: str = 'adam'
    scheduler_type: str = 'reduce_lr'
    batch_size: int = 64
    gradient_clip_norm: float = 1.0
    
    # Fixed parameters (optimized based on previous findings)
    num_pred: int = 12
    num_in: int = 110 
    rid_of_top: int = 4
    n_epochs: int = 300
    time_window: int = 12
    
    # Augmentation parameters
    enable_data_augmentation: bool = False  # Disabled since we have synthetic ARs

def create_scheduler(optimizer, scheduler_type, n_epochs):
    """Create different types of learning rate schedulers"""
    if scheduler_type == 'step_lr':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=n_epochs//10, gamma=0.9)
    elif scheduler_type == 'reduce_lr':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=50
        )
    elif scheduler_type == 'warmup_cosine':
        return get_warmup_cosine_scheduler(optimizer, n_epochs)
    else:  # constant
        return None

def get_warmup_cosine_scheduler(optimizer, n_epochs):
    """Warmup + Cosine Annealing scheduler"""
    warmup_epochs = n_epochs // 10
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch) / float(max(1, warmup_epochs))
        else:
            progress = (epoch - warmup_epochs) / (n_epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate R2 score."""
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
    
    # Create 12 hour window around the first emergence point (24 hours total)
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

def load_all_ar_ids():
    """Load all AR IDs (original + synthetic)"""
    try:
        with open('/mmfs1/project/mx6/sp3463/SAR_EMERGENCE_RESEARCH-main/data/all_ars_list.txt', 'r') as f:
            ar_ids = [int(line.strip()) for line in f if line.strip()]
        print(f"Loaded {len(ar_ids)} ARs (original + synthetic)")
        return ar_ids
    except FileNotFoundError:
        # Fallback to original ARs if synthetic not generated yet
        original_ars = [11130,11149,11158,11162,11199,11327,11344,11387,11393,11416,11422,11455,11619,11640,11660,11678,11682,11765,11768,11776,11916,11928,12036,12051,12085,12089,12144,12175,12203,12257,12331,12494,12659,12778,12864,12877,12900,12929,13004,13085,13098]
        print(f"Using {len(original_ars)} original ARs (synthetic ARs not found)")
        return original_ars

def load_all_ars_data(ARs, rid_of_top, size, num_in, num_pred):
    """Load data for all ARs (original + synthetic)"""
    all_inputs = []
    all_intensities = []
    
    for AR in ARs:
        try:
            # Load from correct locations based on AR ID
            if AR >= 20000:  # Synthetic AR
                ar_dir = f'/mmfs1/project/mx6/sp3463/SAR_EMERGENCE_RESEARCH-main/data/AR{AR}'
            else:  # Original AR
                ar_dir = f'/mmfs1/project/mx6/jst26/SAR_EMERGENCE_RESEARCH/data/AR{AR}'
            
            power_maps = np.load(f'{ar_dir}/mean_pmdop{AR}_flat.npz', allow_pickle=True)
            mag_flux = np.load(f'{ar_dir}/mean_mag{AR}_flat.npz', allow_pickle=True)
            intensities = np.load(f'{ar_dir}/mean_int{AR}_flat.npz', allow_pickle=True)
            
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
            
            print(f"Loaded AR{AR} successfully")
            
        except Exception as e:
            print(f"Failed to load AR{AR}: {e}")
            continue
    
    if len(all_inputs) == 0:
        raise ValueError("No ARs could be loaded!")
    
    all_inputs = np.stack(all_inputs, axis=-1)
    all_intensities = np.stack(all_intensities, axis=-1)
    
    print(f"Per-AR normalization applied (like LSTM)")
    print(f"  Each AR normalized to [0,1] using its own min/max values")
    print(f"  Preserves relative distances within each AR")
    print(f"all_inputs shape: {all_inputs.shape}")
    print(f"all_intensities shape: {all_intensities.shape}")
    
    return all_inputs, all_intensities

def cross_ar_tile_data_preparation(tile, size, all_power_maps, all_intensities, num_in, num_pred, enable_augmentation=False):
    """Prepare data for one tile across all ARs with NO augmentation (we have synthetic ARs now)"""
    
    print(f"  Preparing data for Tile {tile} across all ARs...")
    
    # Combine all AR data for this tile (no augmentation needed with synthetic ARs)
    X_list, y_list = [], []
    
    for ar_idx in range(all_power_maps.shape[-1]):
        power_maps = all_power_maps[:, :, :, ar_idx]
        intensities = all_intensities[:, :, ar_idx]
        
        try:
            X_ar, y_ar = lstm_ready(tile, size, power_maps, intensities, num_in, num_pred)
            if len(X_ar) > 0:
                X_list.append(X_ar)
                y_list.append(y_ar)
        except:
            continue
    
    if len(X_list) > 0:
        X_tile = torch.cat(X_list, dim=0)
        y_tile = torch.cat(y_list, dim=0)
        
        # NO augmentation since we have synthetic ARs providing diversity
        #print(f"    Tile {tile}: {len(X_tile)} samples across all ARs (no augmentation needed)")
    else:
        X_tile = torch.tensor([])
        y_tile = torch.tensor([])
    
    return X_tile, y_tile

def collate_fn_pad(batch):
    inputs, targets = zip(*batch)
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0.0)
    lengths = [len(x) for x in inputs]
    attention_mask = torch.arange(padded_inputs.size(1))[None, :] >= torch.tensor(lengths)[:, None]
    stacked_targets = torch.stack(targets, dim=0)
    return padded_inputs, stacked_targets, attention_mask

def prepare_full_sequence_data(all_inputs, all_intensities, num_pred, min_in=24):
    print(f"Preparing dataset with growing window strategy (min_in={min_in})...")
    all_samples = []
    num_ars = all_inputs.shape[-1]
    num_tiles = all_inputs.shape[0]
    for ar_idx in range(num_ars):
        for tile_idx in range(num_tiles):
            input_series = all_inputs[tile_idx, :, :, ar_idx].T
            intensity_series = all_intensities[tile_idx, :, ar_idx]
            series_len = input_series.shape[0]
            for t_start in range(series_len - num_pred - min_in + 1):
                t_end_input = t_start + min_in
                input_seq = input_series[:t_end_input, :]
                t_start_label = t_end_input
                t_end_label = t_start_label + num_pred
                target_seq = intensity_series[t_start_label:t_end_label]
                if input_seq.shape[0] > 0 and len(target_seq) == num_pred:
                    all_samples.append(
                        (torch.from_numpy(input_seq.copy()).float(), torch.from_numpy(target_seq.copy()).float())
                    )
    print(f"Generated {len(all_samples)} samples from all ARs and tiles.")
    return all_samples

def run_single_configuration(config: HyperparameterConfig, device: torch.device, experiment_idx: int, scheduler_name = None) -> Dict[str, Any]:
    # Log all experiment parameters to stdout and wandb
    print(f"\nExperiment {experiment_idx} parameters:")
    for k, v in asdict(config).items():
        print(f"  {k}: {v}")
    import wandb
    wandb.config.update(asdict(config))
    print(f"--- Running New Configuration ---")
    print(f"Strategy: Full sequence training with variable length inputs.")
    ARs = load_all_ar_ids()
    size = 9
    rid_of_top = config.rid_of_top
    num_pred = config.num_pred
    n_epochs = config.n_epochs
    if scheduler_name is None:
        scheduler_name = config.scheduler_type

    all_inputs, all_intensities = load_all_ars_data(ARs, rid_of_top, size, config.num_in, num_pred)
    all_samples = prepare_full_sequence_data(all_inputs, all_intensities, num_pred, min_in=24)
    if not all_samples:
        print("No samples generated. Aborting run.")
        return {'error': 'No data'}
    train_size = int(0.85 * len(all_samples))
    val_size = len(all_samples) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(all_samples, [train_size, val_size])
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        collate_fn=collate_fn_pad, pin_memory=True, num_workers=4
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        collate_fn=collate_fn_pad, pin_memory=True, num_workers=4
    )
    max_seq_len = all_inputs.shape[2]
    model = SpatioTemporalTransformer(
        input_dim=5,
        max_seq_len=max_seq_len,
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        ff_dim=config.ff_dim,
        num_layers=config.num_layers,
        output_dim=num_pred,
        dropout=config.dropout,
        use_pre_mlp_norm=True,
    ).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = create_scheduler(optimizer, config.scheduler_type, n_epochs)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples.")
    best_val_loss = float('inf')
    for epoch in range(n_epochs):
        model.train()
        epoch_train_loss = 0.0
        for X_batch, y_batch, mask_batch in train_loader:
            X_batch, y_batch, mask_batch = X_batch.to(device), y_batch.to(device), mask_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch, src_key_padding_mask=mask_batch)
            loss = loss_fn(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.gradient_clip_norm)
            optimizer.step()
            epoch_train_loss += loss.item()
        avg_train_loss = epoch_train_loss / len(train_loader)
        model.eval()
        epoch_val_loss = 0.0
        all_val_preds = []
        all_val_trues = []
        with torch.no_grad():
            for X_batch, y_batch, mask_batch in val_loader:
                X_batch, y_batch, mask_batch = X_batch.to(device), y_batch.to(device), mask_batch.to(device)
                outputs = model(X_batch, src_key_padding_mask=mask_batch)
                loss = loss_fn(outputs, y_batch)
                epoch_val_loss += loss.item()
                all_val_preds.append(outputs.cpu().numpy())
                all_val_trues.append(y_batch.cpu().numpy())
        avg_val_loss = epoch_val_loss / len(val_loader)
        all_val_preds_np = np.concatenate(all_val_preds)
        all_val_trues_np = np.concatenate(all_val_trues)
        # Compute overall RMSE
        val_rmse = np.sqrt(np.mean((all_val_trues_np - all_val_preds_np) ** 2))
        # Compute emergence RMSE using calculate_emergence_metrics
        emergence_metrics = calculate_emergence_metrics(all_val_trues_np.flatten(), all_val_preds_np.flatten(), time_step=1.0)
        emergence_rmse = emergence_metrics.get('emergence_rmse', float('nan'))
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}, Val RMSE = {val_rmse:.6f}, Emergence RMSE = {emergence_rmse:.6f}")
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_rmse": val_rmse,
            "emergence_rmse": emergence_rmse,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(avg_val_loss)
        else:
            scheduler.step()
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            models_dir = f'/mmfs1/project/mx6/jst26/SAR_EMERGENCE_RESEARCH/transformer/results/final_models_v7'
            os.makedirs(models_dir, exist_ok=True)
            model_path = os.path.join(models_dir, f"transformer_best_model_exp_{experiment_idx}.pth")
            torch.save(model.state_dict(), model_path)
    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print("\n--- NOTE: Final evaluation plots are disabled ---")
    print("The 'evaluate_models_for_ar' script must be updated to support variable-length inputs.")
    print("Using validation loss as the performance metric for this run.")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    all_preds = []
    all_trues = []
    with torch.no_grad():
        for X_batch, y_batch, mask_batch in val_loader:
            outputs = model(X_batch.to(device), mask_batch.to(device))
            all_preds.append(outputs.cpu().numpy())
            all_trues.append(y_batch.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_trues = np.concatenate(all_trues)
    final_val_rmse = np.sqrt(np.mean((all_trues - all_preds)**2))
    from sklearn.metrics import r2_score
    final_val_r2 = r2_score(all_trues.flatten(), all_preds.flatten())
    print(f"Final Validation Set RMSE: {final_val_rmse:.6f}")
    print(f"Final Validation Set R2: {final_val_r2:.6f}")
    wandb.log({
        'final_val_rmse': final_val_rmse,
        'final_val_r2': final_val_r2,
    })
    
    # === AR Evaluation and WandB Logging (matching Version 5) ===
    lstm_path = "/mmfs1/project/mx6/jst26/SAR_EMERGENCE_RESEARCH/lstm/results/t12_r4_i110_n3_h64_e1000_l0.01.pth"
    test_ars = [11698, 11726, 13165, 13179, 13183]
    successful_ars = []
    failed_ars = []

    # Build transformer_params dict as expected by eval_full_sequence_v6.py
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
        'hidden_size': config.embed_dim,  # match what eval expects
        'learning_rate': config.learning_rate,
        'use_temporal_conv': getattr(config, 'use_temporal_conv', False),
        'max_seq_len': 240
    }
    temp_output_dir = f'/mmfs1/project/mx6/jst26/SAR_EMERGENCE_RESEARCH/transformer/results/final_models_v7/evaluation_plots_exp_{experiment_idx}_num_in_{config.num_in}'
    os.makedirs(temp_output_dir, exist_ok=True)

    for ar in test_ars:
        try:
            plot_path = evaluate_models_for_ar(ar, lstm_path, model_path, transformer_params, temp_output_dir)
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

    print(f"AR evaluations completed: {len(successful_ars)}/{len(test_ars)} successful")
    
    summary_stats = {
        'best_val_loss': best_val_loss,
        'final_val_rmse': final_val_rmse,
        'final_val_r2': final_val_r2,
        'total_samples': len(all_samples),
    }
    return summary_stats

# ========== HYPERPARAMETER GRID SEARCH FUNCTIONS ==========

def generate_hyperparameter_combinations():
    """Generate input sequence ablation study combinations - Version 5 with Temporal Conv."""
    
    # Define search space for input sequence ablation study
    # All parameters are FIXED except num_in which we vary
    search_space = {
        'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1],
        'weight_decay': [0],
        'optimizer_type': ['adam'],
        'num_layers': [2],
        'num_heads': [4],
        'embed_dim': [64],
        'ff_dim': [128],
        'dropout': [0.0, 0.1],
        'scheduler_type': ['reduce_lr'],
        'batch_size': [32],
        'gradient_clip_norm': [1.0],
        'enable_data_augmentation': [False],
        'use_temporal_conv': [True],
        'num_in': [110]
    }
    
    # Generate all combinations
    keys = list(search_space.keys())
    values = list(search_space.values())
    
    combinations = []
    for combination in itertools.product(*values):
        config_dict = dict(zip(keys, combination))
        
        # Validate that embed_dim is divisible by num_heads
        if config_dict['embed_dim'] % config_dict['num_heads'] != 0:
            continue
            
        combinations.append(config_dict)
    
    print(f"Generated {len(combinations)} input sequence ablation study combinations")
    print(f"Testing window lengths: {search_space['num_in']}")
    return combinations

def evaluate_configuration_performance(results: Dict[str, float]) -> float:
    """Evaluate configuration performance with combined score (lower is better)."""
    emergence_rmse = results.get('avg_emergence_rmse', float('inf'))
    overall_rmse = results.get('avg_overall_rmse', float('inf'))
    
    # Combined score (lower is better) - prioritize emergence performance
    score = 0.7 * emergence_rmse + 0.3 * overall_rmse
    return score

def create_experiment_name(config: HyperparameterConfig, idx: int) -> str:
    """Create descriptive experiment name."""
    conv_suffix = "_TempConv" if config.use_temporal_conv else "_NoConv"
    return f"input_sequence_ablation_v5_{idx:04d}_num_in{config.num_in}_l{config.num_layers}h{config.num_heads}_e{config.embed_dim}{conv_suffix}"

def run_hyperparameter_grid_search(device: torch.device, max_experiments: int = None) -> Tuple[HyperparameterConfig, Dict]:
    """Run input sequence ablation study with temporal convolution."""
    
    # Generate all parameter combinations
    param_combinations = generate_hyperparameter_combinations()
    
    if max_experiments is not None:
        param_combinations = param_combinations[:max_experiments]
        print(f"Limited to first {max_experiments} experiments")
    
    print(f"Running input sequence ablation study with {len(param_combinations)} configurations")
    
    best_config = None
    best_score = float('inf')
    all_results = {}
    
    # Track progress
    start_time = time.time()
    
    for idx, params in enumerate(param_combinations):
        print(f"\n{'='*80}")
        print(f"INPUT SEQUENCE ABLATION EXPERIMENT {idx+1}/{len(param_combinations)}")
        print(f"Testing num_in = {params['num_in']}")
        print(f"{'='*80}")
        
        config = HyperparameterConfig(**params)
        
        # Create unique experiment name
        exp_name = create_experiment_name(config, idx)
        
        # Initialize WandB for this experiment
        wandb.init(
            project=f"sar-emergence-comprehensive-full-sequence-v6",
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
                print(f"?? NEW BEST CONFIGURATION! Score: {score:.6f}")
                print(f"   Temporal Conv: {'?' if config.use_temporal_conv else '?'}")
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
            
            print(f"? Experiment {idx+1} completed. Score: {score:.6f}")
            print(f"  Time: {elapsed_time/3600:.1f}h elapsed, {remaining_time/3600:.1f}h remaining")
            print(f"  Best score so far: {best_score:.6f}")
            
        except Exception as e:
            print(f"? Experiment {idx+1} failed: {str(e)}")
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
    results_path = os.path.join(output_dir, 'input_sequence_ablation_v5_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump({
            'best_config': best_config,
            'all_results': all_results,
            'timestamp': datetime.now().isoformat()
        }, f)
    
    # Save best config as text
    best_config_path = os.path.join(output_dir, 'best_input_sequence_ablation_v5_config.txt')
    with open(best_config_path, 'w') as f:
        f.write("BEST INPUT SEQUENCE ABLATION V5 CONFIGURATION\n")
        f.write("="*50 + "\n\n")
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
    summary_path = os.path.join(output_dir, 'input_sequence_ablation_v5_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    
    print(f"\nResults saved to:")
    print(f"  - Full results: {results_path}")
    print(f"  - Best config: {best_config_path}")
    print(f"  - Summary CSV: {summary_path}")

def main():
    """Main function for input sequence ablation study v5."""
    print("?? Starting Input Sequence Ablation Study V5...")
    
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
    
    # Run ablation study
    max_experiments = 5  # Test 5 different window lengths: 60, 90, 110, 116, 130
    best_config, all_results = run_hyperparameter_grid_search(
        device=device,
        max_experiments=max_experiments
    )
    
    print(f"\n{'='*80}")
    print("?? INPUT SEQUENCE ABLATION STUDY V5 COMPLETE!")
    print(f"{'='*80}")
    print(f"Best Configuration:")
    for key, value in asdict(best_config).items():
        print(f"  {key}: {value}")
    
    # Save results
    results_dir = '/mmfs1/project/mx6/jst26/SAR_EMERGENCE_RESEARCH/transformer/results/ablation_study_v5'
    save_results(best_config, all_results, results_dir)
    
    print(f"\n?? Input sequence ablation study v5 completed!")
    print(f"Total configurations tested: {len(all_results)}")
    if all_results:
        print(f"Best score: {min(r['score'] for r in all_results.values()):.6f}")
        best_result = min(all_results.values(), key=lambda x: x['score'])
        print(f"Best window length: {best_result['config'].num_in}")
        print(f"Window lengths tested: {[r['config'].num_in for r in all_results.values()]}")

if __name__ == "__main__":
    main()