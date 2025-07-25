import warnings
warnings.filterwarnings('ignore')
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
from matplotlib import gridspec
from collections import OrderedDict
import re
import matplotlib.gridspec as gridspec
from PIL import Image
import argparse

from transformer.functions import (
    lstm_ready,
    min_max_scaling,
    emergence_indication,
    smooth_with_numpy,
    recalibrate,
    LSTM,
    calculate_extended_metrics
)
from transformer.models.st_transformer import SpatioTemporalTransformer


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


def evaluate_models_for_ar(test_AR, lstm_path, transformer_path, transformer_params, output_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Evaluating AR {test_AR} on: {device}')

    # parse hyperparameters and force rid_of_top
    pat = r't(\d+)_r(\d+)_i(\d+)_n(\d+)_h(\d+)_e(\d+)_l([0-9.]+)\.pth'
    num_pred, _, _, num_layers, hidden_size, n_epochs, lr = (
        int(x) if i!=6 else float(x)
        for i,x in enumerate(re.findall(pat, lstm_path)[0])
    )
    rid_of_top = 1

    # AR settings
    start_tile, before_plot, num_in, NOAA_first, NOAA_second = get_ar_settings(test_AR, rid_of_top)
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

    # load models
    lstm = LSTM(inputs.shape[1],64,3,num_pred).to(device)
    sd = torch.load(lstm_path,map_location=device)
    new_sd = OrderedDict((k[7:] if k.startswith('module.') else k, v) for k,v in sd.items())
    lstm.load_state_dict(new_sd); lstm.eval()

    # Initialize transformer model with correct parameters
    use_pre_mlp_norm = transformer_params['has_pre_mlp_norm'] == 'true'
    trfm = SpatioTemporalTransformer(
        input_dim=inputs.shape[1],
        seq_len=num_in,
        embed_dim=transformer_params['embed_dim'],
        num_heads=transformer_params['num_heads'],
        ff_dim=transformer_params['ff_dim'],
        num_layers=transformer_params['num_layers'],
        output_dim=12,
        dropout=transformer_params['dropout'],
        use_pre_mlp_norm=use_pre_mlp_norm
    ).to(device)
    trfm.load_state_dict(torch.load(transformer_path,map_location=device)); trfm.eval()

    # plotting
    fig = plt.figure(figsize=(16,46))
    fig.subplots_adjust(left=0.15, right=0.85, top=0.97, bottom=0.1)
    gs0 = gridspec.GridSpec(7,1,figure=fig,hspace=.2)
    fut = num_pred-1; thr= -0.01; st=4

    for i in range(7):
        tile_idx = start_tile + i
        disp = tile_idx + 10
        print(f"Tile {disp}")

        X_test, y_test = lstm_ready(tile_idx, size, inputs, ii, num_in, num_pred)
        X_test = X_test.to(device)
        Xt = X_test.view(X_test.size(0), num_in, X_test.size(2))

        with torch.no_grad():
            p_l = lstm(X_test)[:,fut].cpu().numpy()
            p_t = trfm(Xt)[:,fut].cpu().numpy()
        true = y_test[:,fut].numpy()

        last = ii.shape[1]-true.shape[0]-1
        p_l = recalibrate(p_l, ii[tile_idx,last])
        p_t = recalibrate(p_t, ii[tile_idx,last])

        # Calculate metrics for both models
        lstm_metrics = calculate_short_long_metrics(true, p_l)
        trfm_metrics = calculate_short_long_metrics(true, p_t)

        before = ii[tile_idx,last-before_plot:last]
        tcut = time_arr[last-before_plot:last+true.shape[0]]
        tnum = mdates.date2num(tcut)
        nanarr = np.full(before.shape, np.nan)

        # Derivative plots: pad with NaNs for the "before_plot" segment so they align to the full time axis
        d_t = np.gradient(p_t)
        d_l = np.gradient(p_l)
        # create NaN padding for the pre-prediction window
        nan_pad = np.full(before_plot, np.nan)
        # full-length derivative arrays
        d_t_full = np.concatenate([nan_pad, d_t])
        d_l_full = np.concatenate([nan_pad, d_l])

        gs1 = gridspec.GridSpecFromSubplotSpec(5,1,subplot_spec=gs0[i],height_ratios=[18,4,4,4,4],hspace=0.3)

        # All subplots use tnum for x-axis
        ax0 = fig.add_subplot(gs1[0])
        ax0.plot(tnum, np.concatenate((before,true)), 'k-', label='Observed Intensity')
        ax0.plot(tnum, np.concatenate((nanarr,p_l)), 'b-', label='LSTM Prediction')
        ax0.plot(tnum, np.concatenate((nanarr,p_t)), 'r-', label='Transformer Prediction')
        ax0.axvline(NOAA1, color='magenta', linestyle='--', label='NOAA First Record')
        ax0.axvline(NOAA2, color='darkmagenta', linestyle='--', label='NOAA Second Record')
        ax0.set_title(f'Tile {disp}', fontsize=12)
        ax0.set_ylabel('Normalized Intensity', fontsize=9, labelpad=20)
        ax0.set_ylim([-0.1,1.1]); ax0.grid(True)
        ax0.set_yticks([0, 0.25, 0.5, 0.75, 1])
        legend = ax0.legend(bbox_to_anchor=(1.033, .83, 0.223, 0.11), loc='upper left', borderaxespad=0, fontsize=10, framealpha=1, mode='expand')
        legend.get_frame().set_boxstyle('square', pad=1)

        
        # Add metrics table
        create_metrics_table(ax0, lstm_metrics, trfm_metrics)

        ax1 = fig.add_subplot(gs1[1], sharex=ax0)
        d_obs = np.gradient(smooth_with_numpy(np.concatenate((before,true))))
        # dObs/dt: plot full black, overlay limegreen for emergence
        ax1.plot(tnum, d_obs, color='black', linewidth=1)
        ind_o = emergence_indication(d_obs, thr, st)
        for j in range(len(d_obs)-1):
            if ind_o[j] != 0:
                ax1.plot(tnum[j:j+2], d_obs[j:j+2], color='limegreen', linewidth=1)
        ax1.set_ylabel('dObs/dt', fontsize=7, labelpad=10)
        ax1.set_ylim([-0.05,0.05]); ax1.set_yticks([0]); ax1.grid(True)
        ax1.tick_params(labelbottom=False)
        ax1.xaxis.set_major_locator(mdates.DayLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

        ax2 = fig.add_subplot(gs1[2], sharex=ax0)
        # dTrans/dt: plot full red, overlay limegreen for emergence
        ax2.plot(tnum, d_t_full, color='red', linewidth=1)
        ind_t = emergence_indication(d_t_full, thr, st)
        for j in range(len(d_t_full)-1):
            if ind_t[j] != 0:
                ax2.plot(tnum[j:j+2], d_t_full[j:j+2], color='limegreen', linewidth=1)
        ax2.set_ylabel('dTrans/dt', fontsize=7, labelpad=10)
        ax2.set_ylim([-0.05,0.05]); ax2.set_yticks([0]); ax2.grid(True)
        ax2.tick_params(labelbottom=False)
        ax2.xaxis.set_major_locator(mdates.DayLocator())
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2.set_xlim(tnum[0], tnum[-1])

        ax3 = fig.add_subplot(gs1[3], sharex=ax0)
        # dLSTM/dt: plot full blue, overlay limegreen for emergence
        ax3.plot(tnum, d_l_full, color='blue', linewidth=1)
        ind_l = emergence_indication(d_l_full, thr, st)
        for j in range(len(d_l_full)-1):
            if ind_l[j] != 0:
                ax3.plot(tnum[j:j+2], d_l_full[j:j+2], color='limegreen', linewidth=1)
        ax3.set_ylabel('dLSTM/dt', fontsize=7, labelpad=10)
        ax3.set_ylim([-0.05,0.05]); ax3.set_yticks([0]); ax3.grid(True)
        ax3.tick_params(labelbottom=False)
        ax3.set_xlim(tnum[0], tnum[-1])

        # Error curve
        ax4 = fig.add_subplot(gs1[4], sharex=ax0)
        ax4.plot(tnum[before_plot:before_plot+len(true)], np.abs(true - p_l), 'b-')
        ax4.plot(tnum[before_plot:before_plot+len(true)], np.abs(true - p_t), 'r-')
        ax4.axvline(NOAA1, color='magenta', linestyle='--')
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
    trfm_params = extract_params(transformer_path)

    # --- Metrics Table ---
    # Collect metrics for all tiles
    lstm_metrics_list = []
    trfm_metrics_list = []
    for i in range(7):
        # ... (your code for getting predictions: p_l, p_t, true)
        # After you have p_l, p_t, true for each tile:
        lstm_metrics = calculate_extended_metrics(lstm, true, p_l, training_time=0)
        trfm_metrics = calculate_extended_metrics(trfm, true, p_t, training_time=0)
        lstm_metrics_list.append(lstm_metrics)
        trfm_metrics_list.append(trfm_metrics)

    # Calculate mean metrics
    def mean_metric(metrics_list, key):
        vals = [m[key] for m in metrics_list if m[key] is not None]
        return np.mean(vals) if vals else None

    metric_names = ['MAE', 'RMSE', 'R2']
    metrics_table = f"{'Metric':<10} {'LSTM':<12} {'Transformer':<12}\n" + "-"*36 + "\n"
    for name in metric_names:
        lval = mean_metric(lstm_metrics_list, name)
        tval = mean_metric(trfm_metrics_list, name)
        metrics_table += f"{name:<10} {lval:<12.4f} {tval:<12.4f}\n"

    # --- Prepare data ---
    param_headers = ["Parameter", "LSTM", "Transformer"]
    param_rows = [
        ["Time Window", lstm_params['Time Window'], transformer_params['time_window']],
        ["Rid of Top", "1", "1"],
        ["Input Len", lstm_params['Input Len'], num_in],  # Use AR-specific num_in
        ["Layers", lstm_params['Layers'], transformer_params['num_layers']],
        ["Hidden", lstm_params['Hidden'], transformer_params['hidden_size']],
        ["Epochs", lstm_params['Epochs'], "400"],
        ["LR", lstm_params['LR'], transformer_params['learning_rate']],
    ]

    # Metrics
    metric_rows = [
        [name if name != 'R2' else 'R²', f"{mean_metric(lstm_metrics_list, name):.4f}", f"{mean_metric(trfm_metrics_list, name):.4f}"]
        for name in metric_names
    ]

    # --- Summary tables at bottom ---
    metrics_ax = fig.add_axes([0.15, -0.045, 0.3, 0.12])
    metrics_ax.axis('off')

    # Add title for the metrics table
    metrics_ax.text(0.5, 1, 'Overall Performance Metrics', 
                   ha='center', va='center', fontsize=12)

    # Create metrics table
    metrics_data = [['Metric', 'LSTM', 'Transformer']] + metric_rows

    metrics_table = metrics_ax.table(
        cellText=metrics_data,
        colLabels=['Metric', 'LSTM', 'Transformer'],
        colColours=['#e0e0e0'] * 3,
        cellLoc='center',
        loc='upper center'
    )

    # Styling for metrics table
    metrics_table.auto_set_font_size(False)
    metrics_table.set_fontsize(10)
    metrics_table.scale(1, 1.3)

    # Add grid lines, bold header, and alternating row shading for metrics table
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

    # Add grid lines, bold header, and alternating row shading for parameters table
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
    plt.suptitle(f'Model Comparison for AR {test_AR}', y=0.99)
    out = os.path.join(output_dir, f"AR{test_AR}_comparison.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()

    # Crop 100 pixels from the bottom of the saved image
    img = Image.open(out)
    w, h = img.size
    cropped = img.crop((0, 0, w, h - 500))
    cropped.save(out)
    print(f"Comparison plot saved to: {out}")

def calculate_short_long_metrics(true, pred, split_idx=50):
    """Calculate metrics for short-term (first 50 steps) and long-term (remaining steps) predictions."""
    short_true = true[:split_idx]
    short_pred = pred[:split_idx]
    long_true = true[split_idx:]
    long_pred = pred[split_idx:]
    
    def calc_metrics(y_true, y_pred):
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
        return mse, rmse, r2
    
    short_metrics = calc_metrics(short_true, short_pred)
    long_metrics = calc_metrics(long_true, long_pred)
    
    return short_metrics, long_metrics

def create_metrics_table(ax, lstm_metrics, trfm_metrics):
    """Create a metrics table for a tile."""
    short_lstm, long_lstm = lstm_metrics
    short_trfm, long_trfm = trfm_metrics
    
    # Create table data with new layout
    data = [
        ['', 'first 50', 'remaining'],
        ['LSTM', '', ''],
        ['MSE', f'{short_lstm[0]:.4f}', f'{long_lstm[0]:.4f}'],
        ['RMSE', f'{short_lstm[1]:.4f}', f'{long_lstm[1]:.4f}'],
        ['R²', f'{short_lstm[2]:.4f}', f'{long_lstm[2]:.4f}'],
        ['Transformer', '', ''],
        ['MSE', f'{short_trfm[0]:.4f}', f'{long_trfm[0]:.4f}'],
        ['RMSE', f'{short_trfm[1]:.4f}', f'{long_trfm[1]:.4f}'],
        ['R²', f'{short_trfm[2]:.4f}', f'{long_trfm[2]:.4f}']
    ]
    
    # Create table
    table = ax.table(
        cellText=data,
        loc='upper left',
        bbox=[1.02, -0.3, 0.25, 0.6],
        cellLoc='center',
        colLoc='center'
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    
    for pos in ['top', 'bottom', 'left', 'right']:
        table.get_celld()[(0,0)].set_edgecolor('#CCCCCC')
        table.get_celld()[(0,0)].set_linewidth(1)
    
    # Style cells
    for (row, col), cell in table.get_celld().items():
        cell.set_text_props(color='black')
        cell.set_facecolor('white')
        # Add borders to all cells
        cell.set_edgecolor('#CCCCCC')
        cell.set_linewidth(0.5)

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate LSTM and Transformer models')
    
    # Fixed parameters
    parser.add_argument('--time_window', type=int, default=12, help='Time window for predictions')
    parser.add_argument('--num_in', type=int, default=110, help='Input sequence length')
    
    # Transformer model parameters
    parser.add_argument('--num_layers', type=int, required=True, help='Number of transformer layers')
    parser.add_argument('--hidden_size', type=int, required=True, help='Hidden size for transformer')
    parser.add_argument('--learning_rate', type=float, required=True, help='Learning rate for transformer')
    parser.add_argument('--embed_dim', type=int, required=True, help='Embedding dimension for transformer')
    parser.add_argument('--num_heads', type=int, required=True, help='Number of attention heads')
    parser.add_argument('--ff_dim', type=int, required=True, help='Feed-forward dimension')
    parser.add_argument('--dropout', type=float, required=True, help='Dropout rate')
    parser.add_argument('--has_pre_mlp_norm', type=str, required=True, help='Whether model has pre-MLP normalization layer (true/false)')
    
    # Model paths
    parser.add_argument('--lstm_path', type=str, required=True, help='Path to LSTM model checkpoint')
    parser.add_argument('--transformer_path', type=str, required=True, help='Path to transformer model checkpoint')
    
    # Output directory
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save evaluation results')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    # Create transformer parameters dictionary
    transformer_params = {
        'time_window': args.time_window,
        'num_in': args.num_in,
        'num_layers': args.num_layers,
        'hidden_size': args.hidden_size,
        'learning_rate': args.learning_rate,
        'embed_dim': args.embed_dim,
        'num_heads': args.num_heads,
        'ff_dim': args.ff_dim,
        'dropout': args.dropout,
        'has_pre_mlp_norm': args.has_pre_mlp_norm
    }
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    for ar in [11698,11726,13165,13179,13183]:
        evaluate_models_for_ar(ar, args.lstm_path, args.transformer_path, transformer_params, args.output_dir) 