import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json
import re
import os
from collections import OrderedDict
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from matplotlib import gridspec

# Import LSTM functions (from Spyros code)
from functions_spyros import (
    LSTM, lstm_ready, min_max_scaling, smooth_with_numpy, 
    emergence_indication, recalibrate, calculate_metrics,
    find_closest_fits_frame_to_NOAA_record, add_grid_lines, highlight_tile
)

# Import enhanced transformer
from transformer_model import SARTransformerLocalTile

def get_ar_settings_fixed(test_AR, rid_of_top):
    """Get AR-specific settings (same as eval_spyros.py)"""
    settings = {
        11698: (46 - rid_of_top * 9, 50, 96, datetime(2013, 3, 15), datetime(2013, 3, 17)),
        11726: (37 - rid_of_top * 9, 50, 72, datetime(2013, 4, 20), datetime(2013, 4, 22)),
        13165: (28 - rid_of_top * 9, 40, 96, datetime(2022, 12, 12), datetime(2022, 12, 14)),
        13179: (37 - rid_of_top * 9, 40, 96, datetime(2022, 12, 30), datetime(2023, 1, 1)),
        13183: (37 - rid_of_top * 9, 40, 96, datetime(2023, 1, 6), datetime(2023, 1, 8))
    }
    
    if test_AR not in settings:
        raise ValueError(f"Invalid test_AR value: {test_AR}")
    
    return settings[test_AR]

def load_and_preprocess_ar_attention(test_AR, data_path, rid_of_top, size):
    """Load and preprocess AR data for attention transformer"""
    
    # Load data
    power_maps = np.load(f'{data_path}/AR{test_AR}/mean_pmdop{test_AR}_flat.npz', allow_pickle=True)
    mag_flux = np.load(f'{data_path}/AR{test_AR}/mean_mag{test_AR}_flat.npz', allow_pickle=True)
    intensities = np.load(f'{data_path}/AR{test_AR}/mean_int{test_AR}_flat.npz', allow_pickle=True)
    
    power_maps23 = power_maps['arr_0']
    power_maps34 = power_maps['arr_1']
    power_maps45 = power_maps['arr_2']
    power_maps56 = power_maps['arr_3']
    time_arr = power_maps['arr_4']
    mag_flux_data = mag_flux['arr_0']
    intensities_data = intensities['arr_0']
    
    # Trim
    slice_idx = slice(rid_of_top*size, -rid_of_top*size)
    power_maps23 = power_maps23[slice_idx, :]
    power_maps34 = power_maps34[slice_idx, :]
    power_maps45 = power_maps45[slice_idx, :]
    power_maps56 = power_maps56[slice_idx, :]
    mag_flux_data = mag_flux_data[slice_idx, :]
    intensities_data = intensities_data[slice_idx, :]
    
    # Handle NaN
    mag_flux_data[np.isnan(mag_flux_data)] = 0
    intensities_data[np.isnan(intensities_data)] = 0
    
    # Stack and normalize (same as training)
    stacked_maps = np.stack([power_maps23, power_maps34, power_maps45, power_maps56], axis=1)
    stacked_maps[np.isnan(stacked_maps)] = 0
    
    min_p, max_p = np.min(stacked_maps), np.max(stacked_maps)
    min_m, max_m = np.min(mag_flux_data), np.max(mag_flux_data)
    min_i, max_i = np.min(intensities_data), np.max(intensities_data)
    
    stacked_maps = min_max_scaling(stacked_maps, min_p, max_p)
    mag_flux_data = min_max_scaling(mag_flux_data, min_m, max_m)
    intensities_data = min_max_scaling(intensities_data, min_i, max_i)
    
    # Combine features
    mag_flux_reshaped = np.expand_dims(mag_flux_data, axis=1)
    inputs = np.concatenate([stacked_maps, mag_flux_reshaped], axis=1)
    
    return inputs, intensities_data, time_arr

def attention_ready_transformer(tile, size, power_maps, intensities, num_in, num_pred):
    """
    Prepare data for attention transformer - uses overlapping sequences
    """
    # Same preprocessing as LSTM
    final_maps = np.transpose(power_maps, axes=(2, 1, 0))  # (time, features, tiles)
    final_ints = np.transpose(intensities, axes=(1, 0))     # (time, tiles)
    
    X_trans = final_maps[:, :, tile]  # (time, features)
    y_trans = final_ints[:, tile]     # (time,)
    
    # For attention transformer: create overlapping sequences
    total_len = len(X_trans)
    
    if total_len < num_in + num_pred + 20:
        print(f"    Skipping tile {tile}: sequence too short ({total_len} < {num_in + num_pred + 20})")
        return torch.tensor([]), torch.tensor([])
    
    # Create multiple overlapping sequences for evaluation
    step_size = max(1, (total_len - num_in - num_pred) // 5)  # 5 sequences for evaluation
    
    X_list, y_list = [], []
    
    for start_idx in range(0, total_len - num_in - num_pred + 1, step_size):
        end_input = start_idx + num_in
        end_target = end_input + num_pred
        
        input_seq = X_trans[start_idx:end_input]  # (num_in, features)
        target_seq = y_trans[end_input:end_target]  # (num_pred,)
        
        X_list.append(torch.FloatTensor(input_seq))
        y_list.append(torch.FloatTensor(target_seq))
    
    if len(X_list) > 0:
        X_stacked = torch.stack(X_list, dim=0)  # (num_sequences, num_in, features)
        y_stacked = torch.stack(y_list, dim=0)  # (num_sequences, num_pred)
        print(f"    Created {len(X_list)} attention sequences for tile {tile}")
        return X_stacked, y_stacked
    else:
        return torch.tensor([]), torch.tensor([])

def evaluate_lstm_vs_attention_transformer(
    test_AR, 
    lstm_path, 
    transformer_path, 
    data_path, 
    output_dir,
    transformer_config=None
):
    """
    Compare LSTM with enhanced attention transformer
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Evaluating AR {test_AR} on: {device}')
    
    # Get AR settings
    rid_of_top = 1
    size = 9
    start_tile, before_plot, num_in, NOAA_first, NOAA_second = get_ar_settings_fixed(test_AR, rid_of_top)
    
    NOAA1 = mdates.date2num(NOAA_first)
    NOAA2 = mdates.date2num(NOAA_second)
    
    # Load and preprocess data
    inputs, intensities, time_arr = load_and_preprocess_ar_attention(test_AR, data_path, rid_of_top, size)
    input_size = inputs.shape[1]
    
    # Parse LSTM parameters from filename
    lstm_filename = os.path.basename(lstm_path)
    pat = r't(\d+)_r(\d+)_i(\d+)_n(\d+)_h(\d+)_e(\d+)_l([0-9.]+)\.pth'
    match = re.findall(pat, lstm_filename)
    if not match:
        raise ValueError(f"Cannot parse LSTM parameters from: {lstm_filename}")
    
    lstm_num_pred, _, _, lstm_num_layers, lstm_hidden_size, n_epochs, lr = (
        int(x) if i != 6 else float(x) for i, x in enumerate(match[0])
    )
    
    print(f"LSTM params: pred={lstm_num_pred}, layers={lstm_num_layers}, hidden={lstm_hidden_size}")
    
    # Load LSTM model
    lstm = LSTM(input_size, lstm_hidden_size, lstm_num_layers, lstm_num_pred).to(device)
    lstm_state = torch.load(lstm_path, map_location=device)
    new_state_dict = OrderedDict((k[7:] if k.startswith('module.') else k, v) for k, v in lstm_state.items())
    lstm.load_state_dict(new_state_dict)
    lstm.eval()
    
    # Load Enhanced Transformer model
    if transformer_config is None:
        transformer_config = {
            'd_model': 256,
            'nhead': 8,
            'num_layers': 6,
            'dropout': 0.1,
            'output_len': 12
        }
    
    transformer = SARTransformerLocalTile(
        input_dim=input_size,
        d_model=transformer_config['d_model'],
        nhead=transformer_config['nhead'],
        num_layers=transformer_config['num_layers'],
        dropout=transformer_config['dropout'],
        output_len=transformer_config['output_len'],
        max_seq_len=200  # Longer sequences for attention
    ).to(device)
    
    transformer.load_state_dict(torch.load(transformer_path, map_location=device))
    transformer.eval()
    
    print(f"Enhanced Transformer params: d_model={transformer_config['d_model']}, layers={transformer_config['num_layers']}, heads={transformer_config['nhead']}")
    
    # Create comparison plots
    fig = plt.figure(figsize=(16, 46))
    fig.subplots_adjust(left=0.15, right=0.75, top=0.97, bottom=0.1)
    gs0 = gridspec.GridSpec(7, 1, figure=fig, hspace=.2)
    
    lstm_fut = lstm_num_pred - 1
    transformer_fut = transformer_config['output_len'] - 1
    threshold = -0.01
    sust_time = 4
    
    all_lstm_metrics = []
    all_transformer_metrics = []
    
    for i in range(7):
        tile_idx = start_tile + i
        disp = tile_idx + 10
        print(f"Processing Tile {disp}")
        
        # LSTM prediction (sliding window approach)
        X_test_lstm, y_test_lstm = lstm_ready(tile_idx, size, inputs, intensities, num_in, lstm_num_pred)
        X_test_lstm = X_test_lstm.to(device)
        
        with torch.no_grad():
            p_l = lstm(X_test_lstm)[:, lstm_fut].cpu().numpy()
        true = y_test_lstm[:, lstm_fut].numpy()
        
        last = intensities.shape[1] - true.shape[0] - 1
        p_l = recalibrate(p_l, intensities[tile_idx, last])
        
        # Enhanced Transformer prediction (attention-based)
        X_test_trans, y_test_trans = attention_ready_transformer(tile_idx, size, inputs, intensities, 128, transformer_config['output_len'])
        
        if len(X_test_trans) == 0:
            print(f"  Skipping tile {tile_idx} - insufficient data for transformer")
            continue
            
        X_test_trans = X_test_trans.to(device)
        
        with torch.no_grad():
            # Get predictions from all sequences and average them
            p_t_full = transformer(X_test_trans).cpu().numpy()  # Shape: (num_sequences, output_len)
            
            # Average predictions across sequences for robustness
            p_t_avg = np.mean(p_t_full, axis=0)  # Shape: (output_len,)
            p_t = p_t_avg[transformer_fut]  # Single value
            
            print(f"  Transformer: {len(p_t_full)} sequences averaged, prediction: {p_t:.4f}")
        
        # Create transformer prediction array of same length as LSTM
        p_t_array = np.full(len(p_l), p_t)
        
        # Recalibrate
        p_t_array = recalibrate(p_t_array, intensities[tile_idx, last])
        
        # Ensure same length for comparison
        min_len = min(len(p_l), len(p_t_array), len(true))
        p_l = p_l[:min_len]
        p_t_array = p_t_array[:min_len]
        true = true[:min_len]
        
        print(f"  Final shapes - LSTM: {p_l.shape}, Transformer: {p_t_array.shape}, True: {true.shape}")
        print(f"  Prediction ranges - LSTM: [{p_l.min():.4f}, {p_l.max():.4f}], Transformer: [{p_t_array.min():.4f}, {p_t_array.max():.4f}]")
        
        # Calculate metrics
        lstm_metrics = calculate_metrics(true, p_l)
        transformer_metrics = calculate_metrics(true, p_t_array)
        all_lstm_metrics.append(lstm_metrics)
        all_transformer_metrics.append(transformer_metrics)
        
        # Create emergence detection plots
        before = intensities[tile_idx, last-before_plot:last]
        tcut = time_arr[last-before_plot:last+len(true)]
        tnum = mdates.date2num(tcut)
        nanarr = np.full(before.shape, np.nan)
        
        # Calculate derivatives
        d_obs = np.gradient(smooth_with_numpy(np.concatenate((before, true))))
        d_l = np.gradient(p_l)
        d_t = np.gradient(p_t_array)
        
        nan_pad = np.full(before_plot, np.nan)
        d_l_full = np.concatenate([nan_pad, d_l])
        d_t_full = np.concatenate([nan_pad, d_t])
        
        # Emergence indication
        ind_o = emergence_indication(d_obs, threshold, sust_time)
        ind_l = emergence_indication(d_l_full, threshold, sust_time)
        ind_t = emergence_indication(d_t_full, threshold, sust_time)
        
        # Create subplots
        gs1 = gridspec.GridSpecFromSubplotSpec(6, 1, subplot_spec=gs0[i], height_ratios=[18,4,4,4,4,4], hspace=0.3)
        
        # Main intensity plot
        ax0 = fig.add_subplot(gs1[0])
        ax0.plot(tnum, np.concatenate((before, true)), 'k-', label='Observed', linewidth=2)
        ax0.plot(tnum, np.concatenate((nanarr, p_l)), 'b-', label='LSTM (Sliding Window)', linewidth=2)
        ax0.plot(tnum, np.concatenate((nanarr, p_t_array)), 'r-', label='Enhanced Transformer (Attention)', linewidth=2)
        ax0.axvline(NOAA1, color='magenta', linestyle='--', label='NOAA First')
        ax0.axvline(NOAA2, color='darkmagenta', linestyle='--', label='NOAA Second')
        
        ax0.set_title(f'Tile {disp} - LSTM vs Enhanced Attention Transformer', fontsize=12)
        ax0.set_ylabel('Normalized Intensity', fontsize=9)
        ax0.set_ylim([-0.1, 1.1])
        ax0.grid(True)
        ax0.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        ax0.tick_params(labelbottom=False)
        
        # Add metrics comparison
        metrics_text = f"""LSTM (Sliding Window):
            MAE: {lstm_metrics[0]:.4f}
            RMSE: {lstm_metrics[2]:.4f}
            R2: {lstm_metrics[4]:.4f}
          
          Enhanced Transformer (Attention):
            MAE: {transformer_metrics[0]:.4f}
            RMSE: {transformer_metrics[2]:.4f}
            R2: {transformer_metrics[4]:.4f}"""
        
        ax0.text(1.02, 0.5, metrics_text, transform=ax0.transAxes, 
                fontsize=8, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        # Observed derivative
        ax1 = fig.add_subplot(gs1[1], sharex=ax0)
        for j in range(len(d_obs)-1):
            color = 'green' if ind_o[j] != 0 else 'black'
            linewidth = 2 if ind_o[j] != 0 else 1
            ax1.plot(tnum[j:j+2], d_obs[j:j+2], color=color, linewidth=linewidth)
        ax1.axvline(NOAA1, color='magenta', linestyle='--')
        ax1.axvline(NOAA2, color='darkmagenta', linestyle='--')
        ax1.set_ylabel('dObs/dt', fontsize=7)
        ax1.set_ylim([-0.05, 0.05])
        ax1.set_yticks([0])
        ax1.grid(True)
        ax1.tick_params(labelbottom=False)
        
        # Enhanced Transformer derivative
        ax2 = fig.add_subplot(gs1[2], sharex=ax0)
        for j in range(len(d_t_full)-1):
            if not np.isnan(d_t_full[j]):
                color = 'green' if ind_t[j] != 0 else 'red'
                linewidth = 2 if ind_t[j] != 0 else 1
                ax2.plot(tnum[j:j+2], d_t_full[j:j+2], color=color, linewidth=linewidth)
        ax2.axvline(NOAA1, color='magenta', linestyle='--')
        ax2.axvline(NOAA2, color='darkmagenta', linestyle='--')
        ax2.set_ylabel('dTransAttn/dt', fontsize=7)
        ax2.set_ylim([-0.05, 0.05])
        ax2.set_yticks([0])
        ax2.grid(True)
        ax2.tick_params(labelbottom=False)
        
        # LSTM derivative
        ax3 = fig.add_subplot(gs1[3], sharex=ax0)
        for j in range(len(d_l_full)-1):
            if not np.isnan(d_l_full[j]):
                color = 'green' if ind_l[j] != 0 else 'blue'
                linewidth = 2 if ind_l[j] != 0 else 1
                ax3.plot(tnum[j:j+2], d_l_full[j:j+2], color=color, linewidth=linewidth)
        ax3.axvline(NOAA1, color='magenta', linestyle='--')
        ax3.axvline(NOAA2, color='darkmagenta', linestyle='--')
        ax3.set_ylabel('dLSTM/dt', fontsize=7)
        ax3.set_ylim([-0.05, 0.05])
        ax3.set_yticks([0])
        ax3.grid(True)
        ax3.tick_params(labelbottom=False)
        
        # Error comparison
        ax4 = fig.add_subplot(gs1[4], sharex=ax0)
        pred_time = tnum[before_plot:before_plot+len(true)]
        lstm_errors = np.abs(true - p_l)
        transformer_errors = np.abs(true - p_t_array)
        
        ax4.plot(pred_time, lstm_errors, 'b-', label='LSTM Error', linewidth=1)
        ax4.plot(pred_time, transformer_errors, 'r-', label='Enhanced Transformer Error', linewidth=1)
        ax4.axvline(NOAA1, color='magenta', linestyle='--')
        ax4.axvline(NOAA2, color='darkmagenta', linestyle='--')
        ax4.set_ylabel('|Error|', fontsize=7)
        ax4.grid(True)
        ax4.legend(fontsize=6)
        ax4.tick_params(labelbottom=False)
        
        # Prediction comparison
        ax5 = fig.add_subplot(gs1[5], sharex=ax0)
        ax5.plot(pred_time, true, 'k-', label='True', linewidth=2)
        ax5.plot(pred_time, p_l, 'b-', label='LSTM', linewidth=1, alpha=0.8)
        ax5.plot(pred_time, p_t_array, 'r-', label='Enhanced Transformer', linewidth=1, alpha=0.8)
        ax5.axvline(NOAA1, color='magenta', linestyle='--')
        ax5.axvline(NOAA2, color='darkmagenta', linestyle='--')
        ax5.set_ylabel('Predictions', fontsize=7)
        ax5.set_xlabel('Date', fontsize=10)
        ax5.grid(True)
        ax5.legend(fontsize=6)
        ax5.xaxis.set_major_locator(mdates.DayLocator())
        ax5.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax5.tick_params(axis='x', rotation=45)
    
    # Overall summary
    if all_lstm_metrics and all_transformer_metrics:
        avg_lstm_metrics = np.mean(all_lstm_metrics, axis=0)
        avg_transformer_metrics = np.mean(all_transformer_metrics, axis=0)
        
        summary_text = f"""ENHANCED ATTENTION COMPARISON (Average across {len(all_lstm_metrics)} tiles)

        LSTM (Sliding Window):
        - MAE: {avg_lstm_metrics[0]:.4f}
        - RMSE: {avg_lstm_metrics[2]:.4f}  
        - R2: {avg_lstm_metrics[4]:.4f}
        
        Enhanced Transformer (Full Attention):
        - MAE: {avg_transformer_metrics[0]:.4f}
        - RMSE: {avg_transformer_metrics[2]:.4f}
        - R2: {avg_transformer_metrics[4]:.4f}
        
        Improvement:
        - MAE: {((avg_lstm_metrics[0] - avg_transformer_metrics[0])/avg_lstm_metrics[0]*100):+.1f}%
        - RMSE: {((avg_lstm_metrics[2] - avg_transformer_metrics[2])/avg_lstm_metrics[2]*100):+.1f}%
        - R2: {((avg_transformer_metrics[4] - avg_lstm_metrics[4])/avg_lstm_metrics[4]*100):+.1f}%
        
        Enhancements Applied:
        - Multi-head attention with relative positions
        - Multi-scale temporal convolutions
        - Emergence-aware training
        - Weighted attention pooling"""
        
        fig.text(0.77, 0.5, summary_text, fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                verticalalignment='center')
        
        plt.suptitle(f'LSTM vs Enhanced Attention Transformer - AR{test_AR}', y=0.99, fontsize=14)
        
        # Save plot
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        plot_file = output_path / f"AR{test_AR}_LSTM_vs_Enhanced_Attention_Transformer.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Comparison plot saved to: {plot_file}")
        
        return plot_file, {
            'avg_lstm_metrics': avg_lstm_metrics.tolist(),
            'avg_transformer_metrics': avg_transformer_metrics.tolist(),
            'lstm_config': {'sliding_window': True, 'seq_len': num_in},
            'transformer_config': {**transformer_config, 'enhanced_attention': True}
        }
    else:
        print("No metrics computed - all tiles were skipped")
        return None, None

def main():
    parser = argparse.ArgumentParser(description='Enhanced Attention Transformer vs LSTM Comparison')
    parser.add_argument('--data_path', type=str, default='/project/mx6/jst26/SAR_EMERGENCE_RESEARCH/data', help='Data directory')
    parser.add_argument('--lstm_path', type=str, required=True, help='LSTM model path')
    parser.add_argument('--transformer_path', type=str, required=True, help='Enhanced Transformer model path')
    parser.add_argument('--output_dir', type=str, default='/project/mx6/jst26/SAR_EMERGENCE_RESEARCH/sarang_code/comparison_results_attention', help='Output directory')
    parser.add_argument('--test_ars', nargs='+', type=int, default=[13179, 13183, 13165], help='ARs to evaluate')
    parser.add_argument('--config', type=str, help='Transformer config JSON file')
    
    args = parser.parse_args()
    
    # Load transformer config
    transformer_config = None
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            transformer_config = json.load(f)
    
    print("="*70)
    print("ENHANCED ATTENTION TRANSFORMER vs LSTM COMPARISON")
    print("="*70)
    print("LSTM: Sliding window approach")
    print("Enhanced Transformer: Full attention with emergence awareness")
    print("="*70)
    
    all_results = []
    
    for test_ar in args.test_ars:
        print(f"\nEvaluating AR {test_ar}...")
        try:
            plot_path, results = evaluate_lstm_vs_attention_transformer(
                test_ar, args.lstm_path, args.transformer_path, 
                args.data_path, args.output_dir, transformer_config
            )
            
            if results:
                all_results.append({**results, 'test_AR': test_ar})
                print(f"? AR {test_ar} completed successfully")
            else:
                print(f"? AR {test_ar} failed")
        except Exception as e:
            print(f"? AR {test_ar} failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    if all_results:
        print(f"\n{'='*70}")
        print("ENHANCED ATTENTION COMPARISON SUMMARY")
        print(f"{'='*70}")
        
        lstm_scores = [r['avg_lstm_metrics'] for r in all_results]
        trans_scores = [r['avg_transformer_metrics'] for r in all_results]
        
        overall_lstm = np.mean(lstm_scores, axis=0)
        overall_trans = np.mean(trans_scores, axis=0)
        
        print(f"Overall LSTM: MAE={overall_lstm[0]:.4f}, RMSE={overall_lstm[2]:.4f}, R2={overall_lstm[4]:.4f}")
        print(f"Overall Enhanced Transformer: MAE={overall_trans[0]:.4f}, RMSE={overall_trans[2]:.4f}, R2={overall_trans[4]:.4f}")
        
        print(f"\nImprovement (Enhanced Transformer vs LSTM):")
        print(f"  MAE: {((overall_lstm[0] - overall_trans[0])/overall_lstm[0]*100):+.1f}%")
        print(f"  RMSE: {((overall_lstm[2] - overall_trans[2])/overall_lstm[2]*100):+.1f}%")
        print(f"  R2: {((overall_trans[4] - overall_lstm[4])/overall_lstm[4]*100):+.1f}%")

if __name__ == '__main__':
    main()