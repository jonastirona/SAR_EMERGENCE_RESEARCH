import warnings
warnings.filterwarnings('ignore')
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from matplotlib import gridspec
from collections import OrderedDict
import re
import matplotlib.gridspec as gridspec
from PIL import Image
import argparse

from transformer.functions import (
    min_max_scaling,
    emergence_indication,
    smooth_with_numpy,
    recalibrate,
    LSTM,
    calculate_extended_metrics,
    split_sequences
)
# Import NSTransformer components
from functions import NSTransformerWrapper, get_ns_transformer_config


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
        rid_of_top = 1  # Force rid_of_top to 1 for AR 13165
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


def find_first_emergence_window(signal, threshold=-0.01, min_duration=4):
    """Find the first 24-hour emergence window in a signal."""
    emergence_indices = emergence_indication(signal, threshold, min_duration)
    
    # Find the start of the first emergence period
    first_emergence_start = None
    for i, val in enumerate(emergence_indices):
        if val != 0:
            first_emergence_start = i
            break
    
    if first_emergence_start is None:
        return None, None
    
    # Find a 24-hour window starting from first emergence
    window_size = 24
    emergence_end = min(first_emergence_start + window_size, len(signal))
    
    return first_emergence_start, emergence_end


def calculate_emergence_metrics(true, pred_lstm, pred_transformer, time_arr, threshold=-0.01, min_duration=4):
    """Calculate emergence-based metrics including timing differences and window-specific metrics."""
    
    # Calculate derivatives
    d_obs = np.gradient(smooth_with_numpy(true))
    d_lstm = np.gradient(pred_lstm)
    d_transformer = np.gradient(pred_transformer)
    
    # Find first emergence windows
    obs_start, obs_end = find_first_emergence_window(d_obs, threshold, min_duration)
    lstm_start, lstm_end = find_first_emergence_window(d_lstm, threshold, min_duration)
    transformer_start, transformer_end = find_first_emergence_window(d_transformer, threshold, min_duration)
    
    # Calculate metrics for overall series
    def calc_basic_metrics(y_true, y_pred):
        mae = np.mean(np.abs(y_true - y_pred))
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
        return mae, mse, rmse, r2
    
    # Overall metrics
    lstm_mae, lstm_mse, lstm_rmse, lstm_r2 = calc_basic_metrics(true, pred_lstm)
    transformer_mae, transformer_mse, transformer_rmse, transformer_r2 = calc_basic_metrics(true, pred_transformer)
    
    # Calculate emergence window specific metrics if it exists
    lstm_emerg_mae, lstm_emerg_mse, lstm_emerg_rmse, lstm_emerg_r2 = None, None, None, None
    transformer_emerg_mae, transformer_emerg_mse, transformer_emerg_rmse, transformer_emerg_r2 = None, None, None, None
    
    if obs_start is not None and obs_end is not None:
        # Use observed emergence window for both models
        window_true = true[obs_start:obs_end]
        window_lstm = pred_lstm[obs_start:obs_end]
        window_transformer = pred_transformer[obs_start:obs_end]
        
        if len(window_true) > 0:
            lstm_emerg_mae, lstm_emerg_mse, lstm_emerg_rmse, lstm_emerg_r2 = calc_basic_metrics(window_true, window_lstm)
            transformer_emerg_mae, transformer_emerg_mse, transformer_emerg_rmse, transformer_emerg_r2 = calc_basic_metrics(window_true, window_transformer)
    
    # Calculate emergence timing differences (in hours)
    lstm_timing_diff = None
    transformer_timing_diff = None
    
    if obs_start is not None:
        if lstm_start is not None:
            lstm_timing_diff = (lstm_start - obs_start)
        if transformer_start is not None:
            transformer_timing_diff = (transformer_start - obs_start)
    
    return {
        'lstm': {
            'MAE': lstm_mae,
            'MSE': lstm_mse,
            'RMSE': lstm_rmse,
            'R2': lstm_r2,
            'emerg_MAE': lstm_emerg_mae,
            'emerg_MSE': lstm_emerg_mse,
            'emerg_RMSE': lstm_emerg_rmse,
            'emerg_R2': lstm_emerg_r2,
            'emergence_timing_diff': lstm_timing_diff,
            'emergence_window': (lstm_start, lstm_end) if lstm_start is not None else None
        },
        'transformer': {
            'MAE': transformer_mae,
            'MSE': transformer_mse,
            'RMSE': transformer_rmse,
            'R2': transformer_r2,
            'emerg_MAE': transformer_emerg_mae,
            'emerg_MSE': transformer_emerg_mse,
            'emerg_RMSE': transformer_emerg_rmse,
            'emerg_R2': transformer_emerg_r2,
            'emergence_timing_diff': transformer_timing_diff,
            'emergence_window': (transformer_start, transformer_end) if transformer_start is not None else None
        },
        'observed': {
            'emergence_window': (obs_start, obs_end) if obs_start is not None else None
        }
    }


def create_emergence_metrics_table(ax, metrics):
    """Create an emergence-based metrics table for a tile with both overall and window-specific metrics."""
    
    lstm_metrics = metrics['lstm']
    transformer_metrics = metrics['transformer']
    obs_metrics = metrics['observed']
    
    has_emergence_window = obs_metrics['emergence_window'] is not None
    
    data = [['Metric', 'LSTM', 'NSTransformer']]
    
    data.extend([
        ['Overall MAE', f'{lstm_metrics["MAE"]:.4f}', f'{transformer_metrics["MAE"]:.4f}'],
        ['Overall RMSE', f'{lstm_metrics["RMSE"]:.4f}', f'{transformer_metrics["RMSE"]:.4f}'],
        ['Overall R²', f'{lstm_metrics["R2"]:.4f}', f'{transformer_metrics["R2"]:.4f}']
    ])
    
    # Emergence window metrics
    if has_emergence_window:
        data.extend([
            ['Window MAE', 
             f'{lstm_metrics["emerg_MAE"]:.4f}' if lstm_metrics["emerg_MAE"] is not None else 'N/A',
             f'{transformer_metrics["emerg_MAE"]:.4f}' if transformer_metrics["emerg_MAE"] is not None else 'N/A'],
            ['Window RMSE', 
             f'{lstm_metrics["emerg_RMSE"]:.4f}' if lstm_metrics["emerg_RMSE"] is not None else 'N/A',
             f'{transformer_metrics["emerg_RMSE"]:.4f}' if transformer_metrics["emerg_RMSE"] is not None else 'N/A'],
            ['Window R²', 
             f'{lstm_metrics["emerg_R2"]:.4f}' if lstm_metrics["emerg_R2"] is not None else 'N/A',
             f'{transformer_metrics["emerg_R2"]:.4f}' if transformer_metrics["emerg_R2"] is not None else 'N/A']
        ])
    
    # Timing difference
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


def lstm_ready(tile, size, power_maps, intensities, num_in, num_pred, model_seq_len=None):
    X_trans = power_maps[tile]
    y_trans = intensities[tile]
    
    X_trans = X_trans.T
    
    available_time_steps = len(X_trans)

    max_possible_num_in = available_time_steps - num_pred
    
    if max_possible_num_in <= 0:
        raise ValueError(f"Not enough data for tile {tile}. Available: {available_time_steps}, Need at least: {num_pred + 1}")
    
    effective_num_in = min(num_in, max_possible_num_in)
    
    X_ss, y_mm = split_sequences(X_trans, y_trans, effective_num_in, num_pred)
    
    # If model expects a different sequence length, pad accordingly
    target_seq_len = model_seq_len if model_seq_len is not None else effective_num_in
    if effective_num_in < target_seq_len and len(X_ss) > 0:
        padding_length = target_seq_len - effective_num_in
        padding_shape = (len(X_ss), padding_length, X_ss.shape[2])
        padding = np.zeros(padding_shape)
        X_ss = np.concatenate([padding, X_ss], axis=1)
    
    # Convert to tensors
    X = torch.Tensor(X_ss)
    y = torch.Tensor(y_mm)
    
    return X, y


def evaluate_models_for_ar(test_AR, lstm_path, transformer_path, transformer_params, output_dir):
    """Evaluate LSTM and NSTransformer models for a specific AR and return the saved plot path."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Evaluating AR {test_AR} on: {device}')
    
    try:
        print(f"  Debug: transformer_params = {transformer_params}")
        print(f"  Debug: transformer_params keys = {list(transformer_params.keys())}")
        
        # Get model parameters from transformer_params
        num_pred = transformer_params['num_pred']
        num_layers = transformer_params['num_layers'] 
        hidden_size = transformer_params['hidden_size']
        learning_rate = transformer_params['learning_rate']
        print(f"  Debug: Using transformer params - num_pred={num_pred}, num_layers={num_layers}")
        
        if 'rid_of_top' not in transformer_params:
            raise KeyError(f"'rid_of_top' key missing from transformer_params. Available keys: {list(transformer_params.keys())}")
        
        rid_of_top = 1
        print(f"  Debug: rid_of_top = {rid_of_top} (hardcoded for evaluation)")

        # AR settings
        print(f"  Debug: Calling get_ar_settings({test_AR}, {rid_of_top})")
        start_tile, before_plot, num_in, NOAA_first, NOAA_second = get_ar_settings(test_AR, rid_of_top)
        print(f"  Debug: AR settings - start_tile={start_tile}, before_plot={before_plot}, num_in={num_in} (AR-specific)")
        
        NOAA1 = mdates.date2num(NOAA_first)
        NOAA2 = mdates.date2num(NOAA_second)

        # load & prepare data
        base = f'/mmfs1/project/mx6/jst26/SAR_EMERGENCE_RESEARCH/data/AR{test_AR}'
        power = np.load(os.path.join(base, f'mean_pmdop{test_AR}_flat.npz'), allow_pickle=True)
        mag   = np.load(os.path.join(base, f'mean_mag{test_AR}_flat.npz'),   allow_pickle=True)
        cont  = np.load(os.path.join(base, f'mean_int{test_AR}_flat.npz'),   allow_pickle=True)

        pm23, pm34, pm45, pm56, time_arr = (
            power['arr_0'], power['arr_1'], power['arr_2'], power['arr_3'], power['arr_4']
        )
        mf = mag['arr_0']; ii = cont['arr_0']

        # trim
        size = 9
        sl = slice(rid_of_top*size, -rid_of_top*size)
        pm23, pm34, pm45, pm56 = pm23[sl,:], pm34[sl,:], pm45[sl,:], pm56[sl,:]
        mf = mf[sl,:]; ii = ii[sl,:]
        mf[np.isnan(mf)] = 0; ii[np.isnan(ii)] = 0

        # normalize
        stacked = np.stack([pm23,pm34,pm45,pm56],axis=1)
        mp,Mp = stacked.min(), stacked.max()
        mm,Mm = mf.min(), mf.max()
        mi,Mi = ii.min(), ii.max()
        stacked = (stacked - mp)/(Mp-mp)
        mf = (mf - mm)/(Mm-mm)
        ii = (ii - mi)/(Mi-mi)

        # inputs
        inputs = np.concatenate([stacked, np.expand_dims(mf,1)], axis=1)

        # Parse LSTM parameters from LSTM path for LSTM model loading
        pat = r't(\d+)_r(\d+)_i(\d+)_n(\d+)_h(\d+)_e(\d+)_l([0-9.]+)\.pth'
        lstm_num_pred, _, _, lstm_num_layers, lstm_hidden_size, n_epochs, lr = (
            int(x) if i!=6 else float(x)
            for i,x in enumerate(re.findall(pat, lstm_path)[0])
        )
        print(f"  Debug: LSTM params - num_pred={lstm_num_pred}, num_layers={lstm_num_layers}, hidden_size={lstm_hidden_size}")

        # load models
        lstm = LSTM(inputs.shape[1], lstm_hidden_size, lstm_num_layers, lstm_num_pred).to(device)
        sd = torch.load(lstm_path,map_location=device)
        new_sd = OrderedDict((k[7:] if k.startswith('module.') else k, v) for k,v in sd.items())
        lstm.load_state_dict(new_sd); lstm.eval()

        # Load NSTransformer model
        print("  Debug: Loading NSTransformer model")
        
        # Create NSTransformer config
        ns_config = get_ns_transformer_config()
        ns_config.seq_len = num_in
        ns_config.label_len = num_pred
        ns_config.pred_len = num_pred
        ns_config.enc_in = inputs.shape[1]
        ns_config.dec_in = inputs.shape[1]
        ns_config.c_out = 1
        ns_config.d_model = transformer_params['embed_dim']
        ns_config.d_ff = transformer_params['ff_dim']
        ns_config.n_heads = 4
        ns_config.e_layers = transformer_params['num_layers']
        ns_config.d_layers = transformer_params['num_layers']
        ns_config.dropout = transformer_params['dropout']
        
        # Create NSTransformerWrapper and load state dict
        trfm = NSTransformerWrapper(ns_config).to(device)
        transformer_state_dict = torch.load(transformer_path, map_location=device)
        trfm.load_state_dict(transformer_state_dict)
        trfm.eval()

        # Store metrics for all tiles
        all_tile_metrics = []

        # plotting
        fig = plt.figure(figsize=(16,46))
        fig.subplots_adjust(left=0.15, right=0.85, top=0.97, bottom=0.1)
        gs0 = gridspec.GridSpec(7,1,figure=fig,hspace=.2)
        
        lstm_fut = lstm_num_pred-1
        transformer_fut = num_pred-1
        thr= -0.01; st=4

        for i in range(7):
            tile_idx = start_tile + i
            disp = tile_idx + 10
            print(f"Tile {disp}")

            X_test, y_test = lstm_ready(tile_idx, size, inputs, ii, num_in, num_pred)
            X_test = X_test.to(device)
            Xt = X_test.view(X_test.size(0), num_in, X_test.size(2))

            with torch.no_grad():
                p_l = lstm(X_test)[:,lstm_fut].cpu().numpy()
                p_t = trfm(Xt)[:,transformer_fut].cpu().numpy()
            true = y_test[:,lstm_fut].numpy()

            last = ii.shape[1]-true.shape[0]-1
            p_l = recalibrate(p_l, ii[tile_idx,last])
            p_t = recalibrate(p_t, ii[tile_idx,last])

            tile_metrics = calculate_emergence_metrics(true, p_l, p_t, time_arr, thr, st)
            all_tile_metrics.append(tile_metrics)

            before = ii[tile_idx,last-before_plot:last]
            tcut = time_arr[last-before_plot:last+true.shape[0]]
            tnum = mdates.date2num(tcut)
            nanarr = np.full(before.shape, np.nan)

            d_obs = np.gradient(smooth_with_numpy(np.concatenate((before, true))))
            d_l = np.gradient(p_l)
            d_t = np.gradient(p_t)

            nan_pad = np.full(before_plot, np.nan)
            d_l_full = np.concatenate([nan_pad, d_l])
            d_t_full = np.concatenate([nan_pad, d_t])

            # Compute emergence index on d_obs only (unpadded)
            ind_o = emergence_indication(d_obs, thr, st)

            # Define emergence window from first detection in d_obs
            obs_window = None
            for idx, val in enumerate(ind_o):
                if val != 0:
                    start_idx = max(0, idx - 12)
                    end_idx = min(len(d_obs), idx + 12)
                    obs_window = (start_idx, end_idx)
                    break

            # Convert obs_window indices to time values for plotting
            t_start = None
            t_end = None
            if obs_window:
                t_start = tnum[obs_window[0]]
                t_end = tnum[obs_window[1] - 1]

            gs1 = gridspec.GridSpecFromSubplotSpec(5,1,subplot_spec=gs0[i],height_ratios=[18,4,4,4,4],hspace=0.3)

            ax0 = fig.add_subplot(gs1[0])
            ax0.plot(tnum, np.concatenate((before,true)), 'k-', label='Observed Intensity')
            ax0.plot(tnum, np.concatenate((nanarr,p_l)), 'b-', label='LSTM Prediction')
            ax0.plot(tnum, np.concatenate((nanarr,p_t)), 'r-', label='NSTransformer Prediction')
            ax0.axvline(NOAA1, color='magenta', linestyle='--', label='NOAA First Record')
            ax0.axvline(NOAA2, color='darkmagenta', linestyle='--', label='NOAA Second Record')
            
            if obs_window:
                ax0.axvspan(t_start, t_end, color='yellow', alpha=0.3, label='Emergence Window')
            
            ax0.set_title(f'Tile {disp}', fontsize=12)
            ax0.set_ylabel('Normalized Intensity', fontsize=9, labelpad=20)
            ax0.set_ylim([-0.1,1.1]); ax0.grid(True)
            ax0.set_yticks([0, 0.25, 0.5, 0.75, 1])
            legend = ax0.legend(bbox_to_anchor=(1.033, .83, 0.223, 0.11), loc='upper left', borderaxespad=0, fontsize=10, framealpha=1, mode='expand')
            legend.get_frame().set_boxstyle('square', pad=1)
            ax0.tick_params(labelbottom=False)

            create_emergence_metrics_table(ax0, tile_metrics)

            ax1 = fig.add_subplot(gs1[1], sharex=ax0)
            ax1.plot(tnum, d_obs, color='black', linewidth=1)
            
            if obs_window:
                ax1.axvspan(t_start, t_end, color='yellow', alpha=0.3, label='Emergence Window')
            
            for j in range(len(d_obs)-1):
                if ind_o[j] != 0:
                    ax1.plot(tnum[j:j+2], d_obs[j:j+2], color='green', linewidth=1)
            ax1.set_ylabel('dObs/dt', fontsize=7, labelpad=10)
            ax1.set_ylim([-0.05,0.05]); ax1.set_yticks([0]); ax1.grid(True)
            ax1.tick_params(labelbottom=False)

            ax2 = fig.add_subplot(gs1[2], sharex=ax0)
            ax2.plot(tnum, d_t_full, color='red', linewidth=1)
            
            # Add yellow emergence window if it exists
            if obs_window:
                ax2.axvspan(t_start, t_end, color='yellow', alpha=0.3, label='Emergence Window')
            
            ind_t = emergence_indication(d_t_full, thr, st)
            for j in range(len(d_t_full)-1):
                if ind_t[j] != 0:
                    ax2.plot(tnum[j:j+2], d_t_full[j:j+2], color='green', linewidth=1)
            ax2.set_ylabel('dNSTrans/dt', fontsize=7, labelpad=10)
            ax2.set_ylim([-0.05,0.05]); ax2.set_yticks([0]); ax2.grid(True)
            ax2.tick_params(labelbottom=False)
            ax2.set_xlim(tnum[0], tnum[-1])

            ax3 = fig.add_subplot(gs1[3], sharex=ax0)
            ax3.plot(tnum, d_l_full, color='blue', linewidth=1)
            
            # Add yellow emergence window if it exists
            if obs_window:
                ax3.axvspan(t_start, t_end, color='yellow', alpha=0.3, label='Emergence Window')
            
            ind_l = emergence_indication(d_l_full, thr, st)
            for j in range(len(d_l_full)-1):
                if ind_l[j] != 0:
                    ax3.plot(tnum[j:j+2], d_l_full[j:j+2], color='green', linewidth=1)
            ax3.set_ylabel('dLSTM/dt', fontsize=7, labelpad=10)
            ax3.set_ylim([-0.05,0.05]); ax3.set_yticks([0]); ax3.grid(True)
            ax3.tick_params(labelbottom=False)
            ax3.set_xlim(tnum[0], tnum[-1])

            # Error curve
            ax4 = fig.add_subplot(gs1[4], sharex=ax0)
            lstm_errors = np.abs(true - p_l)
            transformer_errors = np.abs(true - p_t)
            
            ax4.plot(tnum[before_plot:before_plot+len(true)], lstm_errors, 'b-', label='LSTM')
            ax4.plot(tnum[before_plot:before_plot+len(true)], transformer_errors, 'r-', label='NSTransformer')
            ax4.axvline(NOAA1, color='magenta', linestyle='--')
            
            if obs_window:
                ax4.axvspan(t_start, t_end, color='yellow', alpha=0.3, label='Emergence Window')
            
            # Add trend lines to error plot
            x_vals = np.arange(len(lstm_errors))
            
            # LSTM error trend
            z_lstm = np.polyfit(x_vals, lstm_errors, 1)
            p_lstm = np.poly1d(z_lstm)
            ax4.plot(tnum[before_plot:before_plot+len(true)], p_lstm(x_vals), 'b--', alpha=0.7, linewidth=1)
            
            # NSTransformer error trend
            z_transformer = np.polyfit(x_vals, transformer_errors, 1)
            p_transformer = np.poly1d(z_transformer)
            ax4.plot(tnum[before_plot:before_plot+len(true)], p_transformer(x_vals), 'r--', alpha=0.7, linewidth=1)
            
            # Add trend line formulas
            slope_lstm, intercept_lstm = z_lstm[0], z_lstm[1]
            slope_transformer, intercept_transformer = z_transformer[0], z_transformer[1]
            
            formula_lstm = f'LSTM: y = {slope_lstm:.4f}x + {intercept_lstm:.4f}'
            formula_transformer = f'NSTransformer: y = {slope_transformer:.4f}x + {intercept_transformer:.4f}'
            
            ax4.text(0.02, 0.98, formula_lstm, transform=ax4.transAxes, fontsize=7, 
                    verticalalignment='top', color='blue', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
            ax4.text(0.02, 0.85, formula_transformer, transform=ax4.transAxes, fontsize=7, 
                    verticalalignment='top', color='red', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
            
            ax4.set_ylabel('|Error|', fontsize=8)
            ax4.set_xlabel('Date', fontsize=10)
            ax4.set_xlim(tnum[0], tnum[-1]); ax4.grid(True)
            ax4.xaxis.set_major_locator(mdates.DayLocator())
            ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax4.tick_params(labelbottom=True)

        
        def extract_params(path):
            fname = os.path.basename(path)
            pat = r't(\d+)_r(\d+)_i(\d+)_n(\d+)_h(\d+)_e(\d+)_l([0-9.]+)\.pth$'
            m = re.search(pat, fname)
            if not m:
                return None
            return {
                'Time Window': m.group(1),
                'Rid of Top': m.group(2),
                'Input Len': m.group(3),
                'Layers': m.group(4),
                'Hidden': m.group(5),
                'Epochs': m.group(6),
                'LR': m.group(7)
            }

        lstm_params = extract_params(lstm_path)
        trfm_params = {
            'Time Window': transformer_params['time_window'],
            'Rid of Top': transformer_params['rid_of_top'], 
            'Input Len': num_in,
            'Layers': transformer_params['num_layers'],
            'Hidden': transformer_params['hidden_size'],
            'Epochs': "400",
            'LR': transformer_params['learning_rate']
        }

        # Calculate mean metrics across all tiles
        def mean_metric(all_metrics, model_key, metric_key):
            values = []
            for tile_metrics in all_metrics:
                val = tile_metrics[model_key][metric_key]
                if val is not None and not np.isnan(val):
                    values.append(val)
            return np.mean(values) if values else None

        # Prepare data for summary tables
        param_headers = ["Parameter", "LSTM", "NSTransformer"]
        param_rows = []
        
        if lstm_params is not None:
            param_rows = [
                ["Time Window", lstm_params['Time Window'], trfm_params['Time Window']],
                ["Rid of Top", lstm_params['Rid of Top'], trfm_params['Rid of Top']],
                ["Input Len", lstm_params['Input Len'], trfm_params['Input Len']],
                ["Layers", lstm_params['Layers'], trfm_params['Layers']],
                ["Hidden", lstm_params['Hidden'], trfm_params['Hidden']],
                ["Epochs", lstm_params['Epochs'], trfm_params['Epochs']],
                ["LR", lstm_params['LR'], trfm_params['LR']],
            ]
        else:
            param_rows = [
                ["Time Window", "N/A", trfm_params['Time Window']],
                ["Rid of Top", "N/A", trfm_params['Rid of Top']],
                ["Input Len", "N/A", trfm_params['Input Len']],
                ["Layers", "N/A", trfm_params['Layers']],
                ["Hidden", "N/A", trfm_params['Hidden']],
                ["Epochs", "N/A", trfm_params['Epochs']],
                ["LR", "N/A", trfm_params['LR']],
            ]

        # Metrics rows with emergence metrics
        metric_names = ['MAE', 'RMSE', 'R2', 'emerg_MAE', 'emerg_RMSE', 'emerg_R2', 'emergence_timing_diff']
        metric_labels = ['Overall MAE', 'Overall RMSE', 'Overall R²', 'Window MAE', 'Window RMSE', 'Window R²', 'Δ Emergence(hrs)']
        
        metric_rows = []
        for name, label in zip(metric_names, metric_labels):
            lstm_val = mean_metric(all_tile_metrics, 'lstm', name)
            transformer_val = mean_metric(all_tile_metrics, 'transformer', name)
            
            if name == 'emergence_timing_diff':
                lstm_str = f"{lstm_val:+.1f}" if lstm_val is not None else "N/A"
                transformer_str = f"{transformer_val:+.1f}" if transformer_val is not None else "N/A"
            else:
                lstm_str = f"{lstm_val:.4f}" if lstm_val is not None else "N/A"
                transformer_str = f"{transformer_val:.4f}" if transformer_val is not None else "N/A"
            
            metric_rows.append([label, lstm_str, transformer_str])

        metrics_ax = fig.add_axes([0.15, -0.045, 0.3, 0.12])
        metrics_ax.axis('off')

        metrics_ax.text(0.5, 1, 'Overall Performance Metrics', 
                       ha='center', va='center', fontsize=12)

        # Create metrics table
        metrics_data = [['Metric', 'LSTM', 'NSTransformer']] + metric_rows

        metrics_table = metrics_ax.table(
            cellText=metrics_data,
            colLabels=['Metric', 'LSTM', 'NSTransformer'],
            colColours=['#e0e0e0'] * 3,
            cellLoc='center',
            loc='upper center'
        )

        # Styling for metrics table
        metrics_table.auto_set_font_size(False)
        metrics_table.set_fontsize(10)
        metrics_table.scale(1, 1.3)

        for (row, col), cell in metrics_table.get_celld().items():
            cell.set_edgecolor('gray')
            cell.set_linewidth(0.5)
            if row == 0:
                cell.set_text_props(weight='bold')
            elif row % 2 == 1:
                cell.set_facecolor('#f9f9f9')
            else:
                cell.set_facecolor('white')

        # Parameters table
        params_ax = fig.add_axes([0.5, -0.045, 0.3, 0.12])
        params_ax.axis('off')

        # Add title for the parameters table
        params_ax.text(0.5, 1, 'Model Parameters', 
                      ha='center', va='center', fontsize=12)

        # Create parameters table
        params_table = params_ax.table(
            cellText=param_rows,
            colLabels=param_headers,
            colColours=['#e0e0e0'] * 3,
            cellLoc='center',
            loc='upper center'
        )

        # Styling for parameters table
        params_table.auto_set_font_size(False)
        params_table.set_fontsize(10)
        params_table.scale(1, 1.3)

        for (row, col), cell in params_table.get_celld().items():
            cell.set_edgecolor('gray')
            cell.set_linewidth(0.5)
            if row == 0:
                cell.set_text_props(weight='bold')
            elif row % 2 == 1:
                cell.set_facecolor('#f9f9f9')
            else:
                cell.set_facecolor('white')

        plt.tight_layout(rect=[0,0,0.8,0.96]); plt.subplots_adjust(right=0.8)
        plt.suptitle(f'Model Comparison for AR {test_AR} (NSTransformer)', y=0.99)
        out = os.path.join(output_dir, f"AR{test_AR}_ns_transformer_comparison.png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.close()

        # Crop 100 pixels from the bottom of the saved image
        img = Image.open(out)
        w, h = img.size
        cropped = img.crop((0, 0, w, h - 500))
        cropped.save(out)
        print(f"Comparison plot saved to: {out}")
        
        # Return the path to the saved image for potential wandb artifact upload
        return out

    except Exception as e:
        import traceback
        print(f"  ERROR in parameter setup for AR {test_AR}:")
        print(f"    Exception type: {type(e).__name__}")
        print(f"    Exception message: {str(e)}")
        print(f"    Full traceback:")
        traceback.print_exc()
        raise


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate LSTM and NSTransformer models with emergence metrics')
    
    # Fixed parameters
    parser.add_argument('--time_window', type=int, default=12, help='Time window for predictions')
    parser.add_argument('--rid_of_top', type=int, default=1, help='Number of top rows to remove')
    parser.add_argument('--num_in', type=int, default=110, help='Input sequence length')
    parser.add_argument('--num_pred', type=int, default=12, help='Number of predictions to make')
    
    # NSTransformer model parameters
    parser.add_argument('--num_layers', type=int, required=True, help='Number of transformer layers')
    parser.add_argument('--hidden_size', type=int, required=True, help='Hidden size for transformer')
    parser.add_argument('--learning_rate', type=float, required=True, help='Learning rate for transformer')
    parser.add_argument('--embed_dim', type=int, required=True, help='Embedding dimension for transformer')
    parser.add_argument('--num_heads', type=int, required=True, help='Number of attention heads')
    parser.add_argument('--ff_dim', type=int, required=True, help='Feed-forward dimension')
    parser.add_argument('--dropout', type=float, required=True, help='Dropout rate')
    
    # Model paths
    parser.add_argument('--lstm_path', type=str, required=True, help='Path to LSTM model checkpoint')
    parser.add_argument('--transformer_path', type=str, required=True, help='Path to NSTransformer model checkpoint')
    
    # Output directory
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save evaluation results')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    # Create transformer parameters dictionary
    transformer_params = {
        'time_window': args.time_window,
        'rid_of_top': args.rid_of_top,
        'num_in': args.num_in,
        'num_pred': args.num_pred,
        'num_layers': args.num_layers,
        'hidden_size': args.hidden_size,
        'learning_rate': args.learning_rate,
        'embed_dim': args.embed_dim,
        'num_heads': args.num_heads,
        'ff_dim': args.ff_dim,
        'dropout': args.dropout
    }
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    for ar in [11698,11726,13165,13179,13183]:
        evaluate_models_for_ar(ar, args.lstm_path, args.transformer_path, transformer_params, args.output_dir)
