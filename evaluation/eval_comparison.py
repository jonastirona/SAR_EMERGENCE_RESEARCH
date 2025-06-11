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


def evaluate_models_for_ar(test_AR, lstm_path, transformer_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Evaluating AR {test_AR} on: {device}')

    # parse hyperparameters and force rid_of_top
    pat = r't(\d+)_r(\d+)_i(\d+)_n(\d+)_h(\d+)_e(\d+)_l([0-9.]+)\.pth'
    num_pred, _, _, num_layers, hidden_size, n_epochs, lr = (
        int(x) if i!=6 else float(x)
        for i,x in enumerate(re.findall(pat, transformer_path)[0])
    )
    rid_of_top = 1

    # AR settings
    start_tile, before_plot, num_in, NOAA_first, NOAA_second = get_ar_settings(test_AR, rid_of_top)
    NOAA1 = mdates.date2num(NOAA_first)
    NOAA2 = mdates.date2num(NOAA_second)

    # load & prepare data
    base = f'/home/jonas/Documents/SAR_EMERGENCE_RESEARCH/data/AR{test_AR}'
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

    trfm = SpatioTemporalTransformer(
        input_dim=inputs.shape[1],
        seq_len=num_in,
        embed_dim=64,
        num_heads=8,
        ff_dim=128,
        num_layers=2,
        output_dim=12,
        dropout=0.1
    ).to(device)
    trfm.load_state_dict(torch.load(transformer_path,map_location=device)); trfm.eval()

    # plotting
    fig = plt.figure(figsize=(16,48))
    fig.subplots_adjust(left=0.15, right=0.85, top=0.95, bottom=0.1)
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
        ["Time Window", lstm_params['Time Window'], trfm_params['Time Window']],
        ["Rid of Top", lstm_params['Rid of Top'], trfm_params['Rid of Top']],
        ["Input Len", lstm_params['Input Len'], trfm_params['Input Len']],
        ["Layers", lstm_params['Layers'], trfm_params['Layers']],
        ["Hidden", lstm_params['Hidden'], trfm_params['Hidden']],
        ["Epochs", lstm_params['Epochs'], trfm_params['Epochs']],
        ["LR", lstm_params['LR'], trfm_params['LR']],
    ]

    # Metrics
    metric_rows = [
        [name, f"{mean_metric(lstm_metrics_list, name):.4f}", f"{mean_metric(trfm_metrics_list, name):.4f}"]
        for name in metric_names
    ]

    table_ax = fig.add_axes([0.3, 0.01, 0.4, 0.07])
    table_ax.axis('off')

    # --- Combine parameter and metric rows ---
    all_headers = param_headers
    all_rows = param_rows + [["", "", ""]] + [["Metric", "LSTM", "Transformer"]] + metric_rows

    # --- Create the table ---
    the_table = table_ax.table(
        cellText=all_rows,
        colLabels=all_headers,
        loc='center',
        cellLoc='center',
        colLoc='center',
        bbox=[0, 0, 1, 1]
    )
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(7)
    the_table.scale(1, 1.1)

    plt.tight_layout(rect=[0,0,0.8,0.96]); plt.subplots_adjust(right=0.8)
    plt.suptitle(f'Model Comparison for AR {test_AR}', y=0.99)
    out = f"/home/jonas/Documents/SAR_EMERGENCE_RESEARCH/evaluation/results/AR{test_AR}_comparison.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
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

if __name__ == '__main__':
    lstm_path = "/home/jonas/Documents/SAR_EMERGENCE_RESEARCH/lstm/results/t12_r4_i110_n3_h64_e1000_l0.01.pth"
    transformer_path = "/home/jonas/Documents/SAR_EMERGENCE_RESEARCH/transformer/results/st_transformer/t12_r4_i110_n3_h64_e400_l0.005.pth"
    for ar in [11698,11726,13165,13179,13183]:
        evaluate_models_for_ar(ar, lstm_path, transformer_path)