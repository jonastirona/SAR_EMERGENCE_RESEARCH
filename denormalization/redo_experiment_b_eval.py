import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import pickle
from pathlib import Path
import argparse
from datetime import datetime, timedelta
import time
import matplotlib.pyplot as plt
import os
import re
from collections import OrderedDict
import matplotlib.dates as mdates
from matplotlib import gridspec
from matplotlib.ticker import MaxNLocator

# Import from experiment_B
import sys
sys.path.append('/project/mx6/jst26/temp/experiment_B')
from transformer_model_B import SARTransformerLocalTile
from data_loader_transformer_B import load_ar_data_enhanced, cross_ar_tile_data_preparation_attention
from functions_spyros import (
    LSTM, lstm_ready, min_max_scaling, smooth_with_numpy, 
    emergence_indication, recalibrate, calculate_metrics
)


def load_tuned_thresholds(thresholds_file):
    """Load tuned emergence detection thresholds"""
    import json
    with open(thresholds_file, 'r') as f:
        return json.load(f)

def get_emergence_threshold(test_AR, tile_idx, thresholds_data, denorm_type):
    """Get the appropriate emergence threshold for the given AR (per-AR, not per-tile)"""
    ar_key = str(test_AR)
    if ar_key not in thresholds_data:
        # Fallback to original threshold if AR not found
        return -0.01
    
    if denorm_type == 'minmax':
        # Use AR-wide threshold (consistent with per-AR normalization)
        return thresholds_data[ar_key].get('minmax_threshold_mean', -0.01)
    elif denorm_type == 'geometric':
        # Use AR-wide threshold (consistent with per-AR normalization)
        return thresholds_data[ar_key].get('geometric_threshold_mean', -0.01)
    else:
        return -0.01

def denormalize_per_ar(normalized_data, min_val, max_val):
    """Denormalize per-AR min-max scaled data back to raw values"""
    if max_val == min_val:
        return np.full_like(normalized_data, min_val)
    return normalized_data * (max_val - min_val) + min_val

def get_ar_settings_fixed(test_AR, rid_of_top):
    """Get AR-specific settings"""
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

def load_and_preprocess_ar_eval_template(test_AR, data_path, rid_of_top, size):
    """Load and preprocess AR data with per-AR normalization (Experiment B style)"""
    base = f'{data_path}/AR{test_AR}'
    power = np.load(os.path.join(base, f'mean_pmdop{test_AR}_flat.npz'), allow_pickle=True)
    mag   = np.load(os.path.join(base, f'mean_mag{test_AR}_flat.npz'),   allow_pickle=True)
    cont  = np.load(os.path.join(base, f'mean_int{test_AR}_flat.npz'),   allow_pickle=True)

    pm23, pm34, pm45, pm56, time_arr = (
        power['arr_0'], power['arr_1'], power['arr_2'], power['arr_3'], power['arr_4']
    )
    mf = mag['arr_0']; ii = cont['arr_0']

    # Store raw data BEFORE trimming for proper denormalization
    ii_raw_full = ii.copy()
    
    sl = slice(rid_of_top*size, -rid_of_top*size)
    pm23, pm34, pm45, pm56 = pm23[sl,:], pm34[sl,:], pm45[sl,:], pm56[sl,:]
    mf = mf[sl,:]; ii = ii[sl,:]
    mf[np.isnan(mf)] = 0; ii[np.isnan(ii)] = 0

    # Use Experiment B normalization approach
    stacked = np.stack([pm23,pm34,pm45,pm56],axis=1)
    stacked[np.isnan(stacked)] = 0
    
    # Per-AR normalization (same as Experiment B)
    min_p, max_p = np.min(stacked), np.max(stacked)
    min_m, max_m = np.min(mf), np.max(mf)
    min_i, max_i = np.min(ii), np.max(ii)
    
    stacked = min_max_scaling(stacked, min_p, max_p)
    mf = min_max_scaling(mf, min_m, max_m)
    ii = min_max_scaling(ii, min_i, max_i)

    # Combine features (5 total: 4 power + 1 flux)
    mag_flux_reshaped = np.expand_dims(mf, axis=1)
    inputs = np.concatenate([stacked, mag_flux_reshaped], axis=1)
    
    # Return both normalized and raw data for denormalization
    return inputs, ii, time_arr, (min_p, max_p, min_m, max_m, min_i, max_i), ii_raw_full

def lstm_ready_eval_template(tile, size, power_maps, intensities, num_in, num_pred, model_seq_len=None):
    """LSTM ready function"""
    final_maps = np.transpose(power_maps, axes=(2, 1, 0))
    final_ints = np.transpose(intensities, axes=(1,0))
    X_trans = final_maps[:,:,tile]
    y_trans = final_ints[:,tile]
    
    available_time_steps = len(X_trans)
    max_possible_num_in = available_time_steps - num_pred
    
    if max_possible_num_in <= 0:
        raise ValueError(f"Not enough data for tile {tile}")
    
    effective_num_in = min(num_in, max_possible_num_in)
    X_ss, y_mm = split_sequences(X_trans, y_trans, effective_num_in, num_pred)
    
    target_seq_len = model_seq_len if model_seq_len is not None else effective_num_in
    if effective_num_in < target_seq_len and len(X_ss) > 0:
        padding_length = target_seq_len - effective_num_in
        padding_shape = (len(X_ss), padding_length, X_ss.shape[2])
        padding = np.zeros(padding_shape)
        X_ss = np.concatenate([padding, X_ss], axis=1)
    
    X = torch.Tensor(X_ss)
    y = torch.Tensor(y_mm)
    return X, y

def split_sequences(input_sequences, output_sequences, n_steps_in, n_steps_out):
    """Split sequences"""
    X, y = list(), list()
    for i in range(len(input_sequences)):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out - 1
        
        if out_end_ix > len(input_sequences):
            break
            
        seq_x = input_sequences[i:end_ix]
        seq_y = output_sequences[end_ix-1:out_end_ix]
        
        X.append(seq_x)
        y.append(seq_y)
        
    return np.array(X), np.array(y)

def safe_gradient(arr, min_length=2):
    """Safely compute gradient, handling short arrays"""
    arr = np.atleast_1d(arr).flatten()
    if len(arr) < min_length:
        return np.zeros_like(arr)
    return np.gradient(arr)

def find_first_emergence_window(signal, threshold=-0.01, min_duration=4):
    """Find the first 24-hour emergence window in a signal."""
    if len(signal) < min_duration:
        return None, None
    
    emergence_indices = emergence_indication(signal, threshold, min_duration)
    
    first_emergence_start = None
    for i, val in enumerate(emergence_indices):
        if val != 0:
            first_emergence_start = i
            break
    
    if first_emergence_start is None:
        return None, None
    
    window_size = 24
    emergence_end = min(first_emergence_start + window_size, len(signal))
    
    return first_emergence_start, emergence_end

def calculate_emergence_timing_normalized(true_norm, pred_lstm_norm, pred_transformer_norm, threshold=-0.01, min_duration=4):
    """Calculate emergence timing using normalized data (for accurate timing)"""
    d_obs = np.gradient(smooth_with_numpy(true_norm))
    d_lstm = np.gradient(pred_lstm_norm)
    d_transformer = np.gradient(pred_transformer_norm)
    
    obs_start, obs_end = find_first_emergence_window(d_obs, threshold, min_duration)
    lstm_start, lstm_end = find_first_emergence_window(d_lstm, threshold, min_duration)
    transformer_start, transformer_end = find_first_emergence_window(d_transformer, threshold, min_duration)
    
    lstm_timing_diff = None
    transformer_timing_diff = None
    
    if obs_start is not None:
        if lstm_start is not None:
            lstm_timing_diff = (lstm_start - obs_start)
        if transformer_start is not None:
            transformer_timing_diff = (transformer_start - obs_start)
    
    return {
        'lstm': {
            'emergence_timing_diff': lstm_timing_diff,
            'emergence_window': (lstm_start, lstm_end) if lstm_start is not None else None
        },
        'transformer': {
            'emergence_timing_diff': transformer_timing_diff,
            'emergence_window': (transformer_start, transformer_end) if transformer_start is not None else None
        },
        'observed': {
            'emergence_window': (obs_start, obs_end) if obs_start is not None else None
        }
    }

def calculate_accuracy_metrics_denormalized(true_raw, pred_lstm_raw, pred_transformer_raw, timing_window):
    """Calculate accuracy metrics using denormalized data"""
    def calc_basic_metrics(y_true, y_pred):
        mae = np.mean(np.abs(y_true - y_pred))
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
        return mae, rmse, r2
    
    lstm_mae, lstm_rmse, lstm_r2 = calc_basic_metrics(true_raw, pred_lstm_raw)
    transformer_mae, transformer_rmse, transformer_r2 = calc_basic_metrics(true_raw, pred_transformer_raw)
    
    lstm_emerg_mae, lstm_emerg_rmse, lstm_emerg_r2 = None, None, None
    transformer_emerg_mae, transformer_emerg_rmse, transformer_emerg_r2 = None, None, None
    
    if timing_window is not None:
        obs_start, obs_end = timing_window
        if obs_start is not None and obs_end is not None:
            window_true = true_raw[obs_start:obs_end]
            window_lstm = pred_lstm_raw[obs_start:obs_end]
            window_transformer = pred_transformer_raw[obs_start:obs_end]
            
            if len(window_true) > 0:
                lstm_emerg_mae, lstm_emerg_rmse, lstm_emerg_r2 = calc_basic_metrics(window_true, window_lstm)
                transformer_emerg_mae, transformer_emerg_rmse, transformer_emerg_r2 = calc_basic_metrics(window_true, window_transformer)
    
    return {
        'lstm': {
            'MAE': lstm_mae,
            'RMSE': lstm_rmse,
            'R2': lstm_r2,
            'emerg_MAE': lstm_emerg_mae,
            'emerg_RMSE': lstm_emerg_rmse,
            'emerg_R2': lstm_emerg_r2
        },
        'transformer': {
            'MAE': transformer_mae,
            'RMSE': transformer_rmse,
            'R2': transformer_r2,
            'emerg_MAE': transformer_emerg_mae,
            'emerg_RMSE': transformer_emerg_rmse,
            'emerg_R2': transformer_emerg_r2
        }
    }

def calculate_emergence_metrics_detailed(true_norm, pred_lstm_norm, pred_transformer_norm, true_raw, pred_lstm_raw, pred_transformer_raw, time_arr, threshold=-0.01, min_duration=4):
    """Calculate emergence metrics with timing from normalized data and accuracy from denormalized data"""
    # Phase 1: Calculate timing on NORMALIZED data (for accurate timing)
    timing_metrics = calculate_emergence_timing_normalized(
        true_norm, pred_lstm_norm, pred_transformer_norm, 
        threshold, min_duration
    )
    
    # Phase 2: Calculate accuracy metrics on DENORMALIZED data
    accuracy_metrics = calculate_accuracy_metrics_denormalized(
        true_raw, pred_lstm_raw, pred_transformer_raw,
        timing_metrics['observed']['emergence_window']
    )
    
    # Combine both
    return {
        'lstm': {
            **timing_metrics['lstm'],
            **accuracy_metrics['lstm']
        },
        'transformer': {
            **timing_metrics['transformer'],
            **accuracy_metrics['transformer']
        },
        'observed': timing_metrics['observed']
    }

def create_emergence_metrics_table(ax, metrics):
    """Create emergence metrics table"""
    lstm_metrics = metrics['lstm']
    transformer_metrics = metrics['transformer']
    obs_metrics = metrics['observed']
    
    has_emergence_window = obs_metrics['emergence_window'] is not None
    
    data = [['Metric', 'LSTM', 'Transformer']]
    
    data.extend([
        ['Overall MAE', f'{lstm_metrics["MAE"]:.4f}', f'{transformer_metrics["MAE"]:.4f}'],
        ['Overall RMSE', f'{lstm_metrics["RMSE"]:.4f}', f'{transformer_metrics["RMSE"]:.4f}'],
        ['Overall R2', f'{lstm_metrics["R2"]:.4f}', f'{transformer_metrics["R2"]:.4f}']
    ])
    
    if has_emergence_window:
        data.extend([
            ['Window MAE', 
             f'{lstm_metrics["emerg_MAE"]:.4f}' if lstm_metrics["emerg_MAE"] is not None else 'N/A',
             f'{transformer_metrics["emerg_MAE"]:.4f}' if transformer_metrics["emerg_MAE"] is not None else 'N/A'],
            ['Window RMSE', 
             f'{lstm_metrics["emerg_RMSE"]:.4f}' if lstm_metrics["emerg_RMSE"] is not None else 'N/A',
             f'{transformer_metrics["emerg_RMSE"]:.4f}' if transformer_metrics["emerg_RMSE"] is not None else 'N/A'],
            ['Window R2', 
             f'{lstm_metrics["emerg_R2"]:.4f}' if lstm_metrics["emerg_R2"] is not None else 'N/A',
             f'{transformer_metrics["emerg_R2"]:.4f}' if transformer_metrics["emerg_R2"] is not None else 'N/A']
        ])
    
    data.append(['Δ Emergence (hrs)', 
                f'{lstm_metrics["emergence_timing_diff"]:+.0f}' if lstm_metrics["emergence_timing_diff"] is not None else 'N/A',
                f'{transformer_metrics["emergence_timing_diff"]:+.0f}' if transformer_metrics["emergence_timing_diff"] is not None else 'N/A'])
    
    table_height = 0.8 if has_emergence_window else 0.6
    table_y_position = -0.6 if has_emergence_window else -0.4
    
    table = ax.table(
        cellText=data,
        loc='upper left',
        bbox=[1.02, table_y_position, 0.25, table_height],
        cellLoc='center',
        colLoc='center'
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    
    for (row, col), cell in table.get_celld().items():
        cell.set_text_props(color='black')
        cell.set_facecolor('white')
        cell.set_edgecolor('#CCCCCC')
        cell.set_linewidth(0.5)
        
        if row == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#e0e0e0')
        elif row <= 3:
            if row % 2 == 0:
                cell.set_facecolor('#f9f9f9')
        elif has_emergence_window and row <= 6:
            if row % 2 == 1:
                cell.set_facecolor('#fff2cc')
            else:
                cell.set_facecolor('#ffe599')
        else:
            cell.set_facecolor('#d9ead3')
        
        if 'Δ Emergence' in str(cell.get_text().get_text()) and col == 0:
            cell.set_text_props(fontsize=6)
        else:
            cell.set_text_props(fontsize=8)

def evaluate_single_ar_with_denorm(test_AR, 
    model_path, 
    config, 
    data_path, 
    lstm_path, 
    output_dir,
    trial_idx,
    thresholds_data=None, 
    denorm_type="minmax"
):
    """Evaluate single AR with denormalization and improved visuals"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"  Evaluating AR {test_AR} with denormalization...")
    
    try:
        rid_of_top = 1
        size = 9
        start_tile, before_plot, num_in, NOAA_first, NOAA_second = get_ar_settings_fixed(test_AR, rid_of_top)
        
        NOAA1 = mdates.date2num(NOAA_first)
        NOAA2 = mdates.date2num(NOAA_second)
        
        # Load data with normalization stats for denormalization
        inputs, ii, time_arr, norm_stats, ii_raw_full = load_and_preprocess_ar_eval_template(test_AR, data_path, rid_of_top, size)
        mp, Mp, mm, Mm, mi, Mi = norm_stats
        
        # Load LSTM model
        pat = r't(\d+)_r(\d+)_i(\d+)_n(\d+)_h(\d+)_e(\d+)_l([0-9.]+)\.pth'
        lstm_num_pred, _, _, lstm_num_layers, lstm_hidden_size, n_epochs, lr = (
            int(x) if i!=6 else float(x)
            for i,x in enumerate(re.findall(pat, lstm_path)[0])
        )
        
        lstm = LSTM(inputs.shape[1], lstm_hidden_size, lstm_num_layers, lstm_num_pred).to(device)
        sd = torch.load(lstm_path,map_location=device)
        new_sd = OrderedDict((k[7:] if k.startswith('module.') else k, v) for k,v in sd.items())
        lstm.load_state_dict(new_sd); lstm.eval()
        
        # Load Enhanced Transformer (Experiment B version)
        transformer = SARTransformerLocalTile(
            input_dim=inputs.shape[1],
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            output_len=config['output_len'],
            max_seq_len=150,
            use_temporal_conv=config.get('use_temporal_conv', True)
        ).to(device)
        
        transformer.load_state_dict(torch.load(model_path, map_location=device))
        transformer.eval()
        
        conv_status = "ENABLED" if config.get('use_temporal_conv', True) else "DISABLED"
        print(f"    Conv1D Temporal Features: {conv_status}")
        
        all_tile_metrics = []
        
        # Collect all derivative values across all tiles for AR-wide range calculation
        all_ar_derivatives = []
        
        # Store references to derivative axes for later range updates
        derivative_axes = []
        
        # Create evaluation plots with improved visuals
        fig = plt.figure(figsize=(16,46))
        fig.subplots_adjust(left=0.15, right=0.85, top=0.97, bottom=0.1)
        gs0 = gridspec.GridSpec(7,1,figure=fig,hspace=.2)
        
        lstm_fut = lstm_num_pred-1
        transformer_fut = config['output_len']-1
        st=4
        
        for i in range(7):
            tile_idx = start_tile + i
            disp = tile_idx + 10
            print(f"    Processing Tile {disp}")
            
            # Get threshold for this AR (per-AR, not per-tile)
            thr = get_emergence_threshold(test_AR, tile_idx, thresholds_data, denorm_type)
            
            X_test, y_test = lstm_ready_eval_template(tile_idx, size, inputs, ii, num_in, config['output_len'], model_seq_len=num_in)
            X_test = X_test.to(device)
            Xt = X_test.view(X_test.size(0), num_in, X_test.size(2))
            
            with torch.no_grad():
                p_l = lstm(X_test)[:,lstm_fut].cpu().numpy()
                p_t = transformer(Xt)[:,transformer_fut].cpu().numpy()
            true = y_test[:,lstm_fut].numpy()
            
            # DENORMALIZATION: Convert predictions to raw values BEFORE recalibration
            p_l_raw = denormalize_per_ar(p_l, mi, Mi)
            p_t_raw = denormalize_per_ar(p_t, mi, Mi)
            true_raw = denormalize_per_ar(true, mi, Mi)
            
            # Get raw data for recalibration
            last = ii.shape[1]-true.shape[0]-1
            # Use raw data for recalibration point
            recal_point_raw = ii_raw_full[tile_idx, last]
            
            # Apply recalibration to raw predictions
            p_l_raw = recalibrate(p_l_raw, recal_point_raw)
            p_t_raw = recalibrate(p_t_raw, recal_point_raw)
            
            # Calculate metrics with timing from normalized data and accuracy from denormalized data
            # Use fixed threshold=-0.01 for timing calculations
            tile_metrics = calculate_emergence_metrics_detailed(
                true, p_l, p_t,  # normalized data for timing
                true_raw, p_l_raw, p_t_raw,  # denormalized data for accuracy
                time_arr, threshold=-0.01, min_duration=st
            )
            all_tile_metrics.append(tile_metrics)
            
            # Get raw "before" data directly from raw data
            before_raw = ii_raw_full[tile_idx, last-before_plot:last]
            tcut = time_arr[last-before_plot:last+true.shape[0]]
            tnum = mdates.date2num(tcut)
            nanarr = np.full(before_raw.shape, np.nan)
            
            # Calculate derivatives on NORMALIZED data for timing detection
            before_norm = ii[tile_idx, last-before_plot:last]
            d_obs_norm = safe_gradient(smooth_with_numpy(np.concatenate((before_norm, true))))
            d_l_norm = safe_gradient(p_l)
            d_t_norm = safe_gradient(p_t)
            
            # Calculate derivatives on DENORMALIZED data for visualization
            d_obs_raw = safe_gradient(smooth_with_numpy(np.concatenate((before_raw, true_raw))))
            d_l_raw = safe_gradient(p_l_raw)
            d_t_raw = safe_gradient(p_t_raw)
            
            nan_pad = np.full(before_plot, np.nan)
            d_l_full = np.concatenate([nan_pad, d_l_raw])
            d_t_full = np.concatenate([nan_pad, d_t_raw])
            
            # Collect derivative values for AR-wide range calculation
            all_ar_derivatives.extend(d_obs_raw[np.isfinite(d_obs_raw)])
            all_ar_derivatives.extend(d_l_raw[np.isfinite(d_l_raw)])
            all_ar_derivatives.extend(d_t_raw[np.isfinite(d_t_raw)])
            
            # Use normalized timing for emergence window visualization
            obs_window = tile_metrics['observed']['emergence_window']
            
            # Calculate emergence indication using NORMALIZED data (for accurate timing)
            ind_o_norm = emergence_indication(d_obs_norm, -0.01, st)
            ind_l_norm = emergence_indication(d_l_norm, -0.01, st)
            ind_t_norm = emergence_indication(d_t_norm, -0.01, st)
            
            t_start = None
            t_end = None
            if obs_window:
                # Map normalized timing indices to denormalized time array
                # obs_window indices are relative to the normalized data (without 'before' data)
                # Need to add before_plot offset to map to full time array
                start_idx = obs_window[0] + before_plot
                end_idx = obs_window[1] + before_plot - 1
                if start_idx < len(tnum) and end_idx < len(tnum):
                    t_start = tnum[start_idx]
                    t_end = tnum[end_idx]
            
            gs1 = gridspec.GridSpecFromSubplotSpec(5,1,subplot_spec=gs0[i],height_ratios=[18,4,4,4,4],hspace=0.3)
            
            # Main plot with raw intensity values
            ax0 = fig.add_subplot(gs1[0])
            ax0.plot(tnum, np.concatenate((before_raw,true_raw)), 'k-', label='Observed Intensity')
            ax0.plot(tnum, np.concatenate((nanarr,p_l_raw)), 'b-', label='LSTM Prediction')
            
            transformer_label = f'Transformer Prediction'
            ax0.plot(tnum, np.concatenate((nanarr,p_t_raw)), 'r-', label=transformer_label)
            ax0.axvline(NOAA1, color='magenta', linestyle='--', label='NOAA First Record')
            ax0.axvline(NOAA2, color='darkmagenta', linestyle='--', label='NOAA Second Record')
            
            if obs_window:
                ax0.axvspan(t_start, t_end, color='yellow', alpha=0.3, label='Emergence Window')
            
            ax0.set_title(f'Tile {disp} - Trial {trial_idx} - AR {test_AR}', fontsize=12)
            ax0.set_ylabel('Continuum Intensity', fontsize=9, labelpad=20)
            
            # Use consistent AR-wide range for all tiles (better for comparison)
            ar_min, ar_max = np.nanmin(ii_raw_full), np.nanmax(ii_raw_full)
            pad = max(0.05 * (ar_max - ar_min), 0.05)
            ax0.set_ylim([ar_min - pad, ar_max + pad])
            
            ax0.grid(True)
            ax0.yaxis.set_major_locator(MaxNLocator(nbins=8))
            legend = ax0.legend(bbox_to_anchor=(1.033, .83, 0.223, 0.11), loc='upper left', borderaxespad=0, fontsize=10, framealpha=1, mode='expand')
            legend.get_frame().set_boxstyle('square', pad=1)
            ax0.tick_params(labelbottom=False)
            
            create_emergence_metrics_table(ax0, tile_metrics)
            
            # Derivative plots - display denormalized values but use normalized timing
            ax1 = fig.add_subplot(gs1[1], sharex=ax0)
            ax1.plot(tnum, d_obs_raw, color='black', linewidth=1)
            
            if obs_window:
                ax1.axvspan(t_start, t_end, color='yellow', alpha=0.3)
            
            # Use normalized timing indices for highlighting
            for j in range(len(d_obs_norm)-1):
                if ind_o_norm[j] != 0:
                    ax1.plot(tnum[j:j+2], d_obs_raw[j:j+2], color='green', linewidth=1)
            ax1.set_ylabel('dObs/dt', fontsize=7, labelpad=10)
            # Calculate AR-wide maximum range (will be applied after all tiles are processed)
            # For now, set a temporary range that will be updated later
            ax1.set_ylim([-0.05, 0.05])
            ax1.set_yticks([-0.05, 0, 0.05])
            ax1.grid(True)
            ax1.tick_params(labelbottom=False)
            
            ax2 = fig.add_subplot(gs1[2], sharex=ax0)
            ax2.plot(tnum, d_t_full, color='red', linewidth=1)
            
            if obs_window:
                ax2.axvspan(t_start, t_end, color='yellow', alpha=0.3)
            
            # Use normalized timing indices for highlighting (with proper padding alignment)
            for j in range(len(d_t_norm)-1):
                if ind_t_norm[j] != 0:
                    # Map normalized index to denormalized index (accounting for padding)
                    denorm_idx = j + before_plot
                    if denorm_idx < len(d_t_full) - 1:
                        ax2.plot(tnum[denorm_idx:denorm_idx+2], d_t_full[denorm_idx:denorm_idx+2], color='green', linewidth=1)
            ax2.set_ylabel('dTrans/dt', fontsize=7, labelpad=10)
            # Use same y-limits as ax1 (will be updated after all tiles are processed)
            ax2.set_ylim(ax1.get_ylim())
            ax2.set_yticks(ax1.get_yticks())
            ax2.grid(True)
            ax2.tick_params(labelbottom=False)
            ax2.set_xlim(tnum[0], tnum[-1])
            
            ax3 = fig.add_subplot(gs1[3], sharex=ax0)
            ax3.plot(tnum, d_l_full, color='blue', linewidth=1)
            
            if obs_window:
                ax3.axvspan(t_start, t_end, color='yellow', alpha=0.3)
            
            # Use normalized timing indices for highlighting (with proper padding alignment)
            for j in range(len(d_l_norm)-1):
                if ind_l_norm[j] != 0:
                    # Map normalized index to denormalized index (accounting for padding)
                    denorm_idx = j + before_plot
                    if denorm_idx < len(d_l_full) - 1:
                        ax3.plot(tnum[denorm_idx:denorm_idx+2], d_l_full[denorm_idx:denorm_idx+2], color='green', linewidth=1)
            ax3.set_ylabel('dLSTM/dt', fontsize=7, labelpad=10)
            # Use same y-limits as ax1 (will be updated after all tiles are processed)
            ax3.set_ylim(ax1.get_ylim())
            ax3.set_yticks(ax1.get_yticks())
            ax3.grid(True)
            ax3.tick_params(labelbottom=False)
            ax3.set_xlim(tnum[0], tnum[-1])
            
            # Store derivative axes references for later range updates
            derivative_axes.append((ax1, ax2, ax3))
            
            # Error analysis
            ax4 = fig.add_subplot(gs1[4], sharex=ax0)
            lstm_errors = np.abs(true_raw - p_l_raw)
            transformer_errors = np.abs(true_raw - p_t_raw)
            
            ax4.plot(tnum[before_plot:before_plot+len(true_raw)], lstm_errors, 'b-', label='LSTM')
            ax4.plot(tnum[before_plot:before_plot+len(true_raw)], transformer_errors, 'r-', label='Transformer')
            ax4.axvline(NOAA1, color='magenta', linestyle='--')
            
            if obs_window:
                ax4.axvspan(t_start, t_end, color='yellow', alpha=0.3)
            
            ax4.set_ylabel('|Error|', fontsize=8)
            ax4.set_xlabel('Date', fontsize=10)
            ax4.set_xlim(tnum[0], tnum[-1]); ax4.grid(True)
            ax4.xaxis.set_major_locator(mdates.DayLocator())
            ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax4.tick_params(labelbottom=True)
        
        # Apply AR-wide derivative range to all derivative plots
        if len(all_ar_derivatives) > 0:
            max_range = max(abs(np.min(all_ar_derivatives)), abs(np.max(all_ar_derivatives)))
            y_lim = max_range * 1.1  # Add 10% padding
            
            # Update all derivative plots with AR-wide range using stored axes references
            for ax1, ax2, ax3 in derivative_axes:
                ax1.set_ylim([-y_lim, y_lim])
                ax1.set_yticks([-y_lim, 0, y_lim])
                ax2.set_ylim([-y_lim, y_lim])
                ax2.set_yticks([-y_lim, 0, y_lim])
                ax3.set_ylim([-y_lim, y_lim])
                ax3.set_yticks([-y_lim, 0, y_lim])
        
        plt.tight_layout(rect=[0,0,0.8,0.96]); plt.subplots_adjust(right=0.8)
        plt.suptitle(f'Trial {trial_idx} Model Comparison - AR {test_AR} (Conv1D {conv_status})', y=0.99)
        
        # Save as PDF
        plot_dir = Path(output_dir) / 'denormalized_ar_evaluations'
        plot_dir.mkdir(parents=True, exist_ok=True)
        out = plot_dir / f"Trial_{trial_idx:03d}_AR{test_AR}_eval.pdf"
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    ✓ AR {test_AR} evaluation completed with denormalization")
        return str(out)
        
    except Exception as e:
        print(f"    ✗ AR {test_AR} evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_trial_config(trial_idx, output_dir):
    """Load configuration for a specific trial from the CSV results file"""
    csv_path = output_dir / 'Conv1D_search_results_ALL_ARS.csv'
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Results CSV not found: {csv_path}")
    
    # Read CSV and find the row for this trial
    df = pd.read_csv(csv_path)
    trial_row = df[df['trial_idx'] == trial_idx]
    
    if trial_row.empty:
        raise ValueError(f"Trial {trial_idx} not found in results CSV")
    
    # Extract configuration parameters (Experiment B has simpler config)
    config = {
        'd_model': int(trial_row['d_model'].iloc[0]),
        'nhead': int(trial_row['nhead'].iloc[0]),
        'num_layers': int(trial_row['num_layers'].iloc[0]),
        'dropout': float(trial_row['dropout'].iloc[0]),
        'learning_rate': float(trial_row['learning_rate'].iloc[0]),
        'output_len': int(trial_row['output_len'].iloc[0]),
        'use_temporal_conv': bool(trial_row['use_temporal_conv'].iloc[0])
    }
    
    return config

def main():
    parser = argparse.ArgumentParser(description='Redo Experiment B AR Evaluation with Denormalization')
    parser.add_argument('--data_path', type=str, default='/mmfs1/project/mx6/jst26/final/data', help='Data directory')
    parser.add_argument('--output_dir', type=str, default='/project/mx6/jst26/final/experiments/experiment_b_full', help='Output directory')
    parser.add_argument('--lstm_path', type=str, default='/mmfs1/project/mx6/jst26/SAR_EMERGENCE_RESEARCH/lstm/results/t12_r4_i110_n3_h64_e1000_l0.01.pth', help='LSTM model path for comparison')
    parser.add_argument('--trial_range', type=str, default='0-15', help='Trial range to process (e.g., 0-15 or 13-13)')
    parser.add_argument('--test_ars', nargs='+', type=int, default=[11698, 11726, 13165, 13179, 13183], help='ARs to evaluate')
    
    # Load tuned thresholds
    thresholds_file = Path('/project/mx6/jst26/final/threshold_tuning/tuned_thresholds.json')
    thresholds_data = None
    if thresholds_file.exists():
        thresholds_data = load_tuned_thresholds(thresholds_file)
        print(f"Loaded tuned thresholds from {thresholds_file}")
    else:
        print(f"Warning: No tuned thresholds found at {thresholds_file}, using default -0.01")
    

    args = parser.parse_args()
    
    # Parse trial range
    start_trial, end_trial = map(int, args.trial_range.split('-'))
    
    print("="*100)
    print("REDO EXPERIMENT B AR EVALUATION WITH DENORMALIZATION")
    print("="*100)
    print(f"Output directory: {args.output_dir}")
    print(f"Trial range: {start_trial}-{end_trial}")
    print(f"Test ARs: {args.test_ars}")
    print("="*100)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir).resolve()
    
    successful_evaluations = 0
    total_evaluations = 0
    
    for trial_idx in range(start_trial, end_trial + 1):
        print(f"\n{'='*80}")
        print(f"PROCESSING TRIAL {trial_idx}")
        print(f"{'='*80}")
        
        trial_dir = output_dir / f'models/trial_{trial_idx:03d}'
        model_path = trial_dir / 'model.pth'
        
        if not model_path.exists():
            print(f"  ✗ Model not found: {model_path}")
            continue
        
        # Load trial-specific configuration from CSV
        try:
            config = load_trial_config(trial_idx, output_dir)
            print(f"  Config: {config}")
        except Exception as e:
            print(f"  ✗ Failed to load config for trial {trial_idx}: {e}")
            continue
        
        for test_AR in args.test_ars:
            print(f"\n  Evaluating AR {test_AR}...")
            total_evaluations += 1
            
            plot_path = evaluate_single_ar_with_denorm(test_AR,
                str(model_path),
                config,
                args.data_path,
                args.lstm_path,
                str(output_dir),
                trial_idx,
                thresholds_data,
                "minmax"
            )
            
            if plot_path:
                successful_evaluations += 1
                print(f"  ✓ Saved: {plot_path}")
            else:
                print(f"  ✗ Failed to generate plot for AR {test_AR}")
    
    print(f"\n{'='*100}")
    print("EVALUATION COMPLETED!")
    print(f"{'='*100}")
    print(f"Successful evaluations: {successful_evaluations}/{total_evaluations}")
    success_rate = successful_evaluations/total_evaluations*100 if total_evaluations > 0 else 0.0
    print(f"Success rate: {success_rate:.1f}%")
    print(f"PDF outputs saved to: {output_dir}/denormalized_ar_evaluations/")

if __name__ == '__main__':
    main()
