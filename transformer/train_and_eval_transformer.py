import argparse
import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from tqdm import trange
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.dates as mdates
from datetime import datetime
from collections import OrderedDict
import json
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import math
import matplotlib.gridspec as gridspec
from models.st_transformer import SpatioTemporalTransformer

# Suppress warnings
warnings.filterwarnings('ignore')

# Fixed parameters
NUM_PRED = 12
RID_OF_TOP = 4
NUM_IN = 110
EPOCHS = 400
SIZE = 9
TILES = SIZE**2 - 2*SIZE*RID_OF_TOP

# Training ARs
TRAIN_ARs = [11130,11149,11158,11162,11199,11327,11344,11387,11393,11416,11422,11455,11619,11640,11660,11678,11682,11765,11768,11776,11916,11928,12036,12051,12085,12089,12144,12175,12203,12257,12331,12494,12659,12778,12864,12877,12900,12929,13004,13085,13098]

# Test ARs
TEST_ARs = [11698, 11726, 13165, 13179, 13183]

def parse_args():
    parser = argparse.ArgumentParser(description='Train and evaluate SpatioTemporalTransformer')
    parser.add_argument('--dropout', type=float, required=True, help='Dropout rate')
    parser.add_argument('--hidden_size', type=int, required=True, help='Hidden size')
    parser.add_argument('--num_layers', type=int, required=True, help='Number of layers')
    parser.add_argument('--learning_rate', type=float, required=True, help='Learning rate')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Warmup ratio (default: 0.1)')
    parser.add_argument('--ff_ratio', type=float, required=True, help='Feed-forward ratio')
    parser.add_argument('--num_heads', type=int, required=True, help='Number of attention heads')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory path')
    return parser.parse_args()

def get_output_folder(args):
    """Create output folder name based on hyperparameters."""
    folder_name = (
        f"t{NUM_PRED}_"
        f"r{RID_OF_TOP}_"
        f"i{NUM_IN}_"
        f"n{args.num_layers}_"
        f"h{args.hidden_size}_"
        f"e{EPOCHS}_"
        f"l{args.learning_rate:.3f}_"
        f"d{args.dropout:.2f}_"
        f"w{args.warmup_ratio:.2f}_"
        f"f{args.ff_ratio:.1f}_"
        f"nh{args.num_heads}"
    )
    return os.path.join(args.output_dir, folder_name)

def min_max_scaling(arr, min_val, max_val):
    """Normalize array to [0,1] range."""
    return (arr - min_val) / (max_val - min_val)

def split_sequences(input_sequences, output_sequences, n_steps_in, n_steps_out):
    """Split sequences into input/output pairs."""
    X, y = list(), list()
    for i in range(len(input_sequences)):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out - 1
        if out_end_ix > len(input_sequences):
            break
        seq_x, seq_y = input_sequences[i:end_ix], output_sequences[end_ix-1:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def lstm_ready(tile, size, power_maps, intensities, num_in, num_pred):
    """Prepare data for a specific tile."""
    final_maps = np.transpose(power_maps, axes=(2, 1, 0))
    final_ints = np.transpose(intensities, axes=(1,0))
    X_trans = final_maps[:,:,tile]
    y_trans = final_ints[:,tile]
    X_ss, y_mm = split_sequences(X_trans, y_trans, num_in, num_pred)
    return torch.Tensor(X_ss), torch.Tensor(y_mm)

def smooth_with_numpy(d_true, window_size=5):
    """Smooth data using moving average."""
    if window_size <= 1:
        return d_true
    pad_width = window_size // 2
    padded_d_true = np.pad(d_true, pad_width, mode='edge')
    window = np.ones(window_size) / window_size
    smoothed_d_true = np.convolve(padded_d_true, window, mode='same')
    return smoothed_d_true[pad_width:-pad_width] if pad_width else smoothed_d_true

def emergence_indication(d_true, threshold=-0.01, sust_time=4):
    """Detect emergence points in derivative."""
    d_true = smooth_with_numpy(d_true)
    indicator = np.zeros(d_true.shape)
    for j in range(len(d_true)):
        if d_true[j] <= threshold:
            indicator[j] = 1
    sustained = True
    if sustained:
        start_idx = None
        for i in range(len(indicator)):
            if indicator[i] == 1 and start_idx is None:
                start_idx = i
            elif indicator[i] == 0 and start_idx is not None:
                if i - start_idx < sust_time:
                    indicator[start_idx:i] = 0
                start_idx = None
        if start_idx is not None and len(indicator) - start_idx < sust_time:
            indicator[start_idx:] = 0
    return indicator

def recalibrate(pred, previous_value):
    """Recalibrate predictions based on previous value."""
    trend = pred - pred[0]
    new_pred = trend + previous_value
    return new_pred 

def calculate_short_long_metrics(true, pred, split_idx=50):
    """Calculate metrics for short-term (first 50 steps) and long-term (remaining steps) predictions."""
    short_true = true[:split_idx]
    short_pred = pred[:split_idx]
    long_true = true[split_idx:]
    long_pred = pred[split_idx:]
    
    def calc_metrics(y_true, y_pred):
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))
        r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
        return mse, rmse, mae, r2
    
    short_metrics = calc_metrics(short_true, short_pred)
    long_metrics = calc_metrics(long_true, long_pred)
    
    return short_metrics, long_metrics

def create_metrics_table(ax, metrics):
    """Create a metrics table for a tile showing short-term vs long-term performance."""
    short_metrics, long_metrics = metrics
    
    # Create table data with new layout
    data = [
        ['', 'first 50', 'remaining'],
        ['MSE', f'{short_metrics[0]:.4f}', f'{long_metrics[0]:.4f}'],
        ['RMSE', f'{short_metrics[1]:.4f}', f'{long_metrics[1]:.4f}'],
        ['MAE', f'{short_metrics[2]:.4f}', f'{long_metrics[2]:.4f}'],
        ['R²', f'{short_metrics[3]:.4f}', f'{long_metrics[3]:.4f}']
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
        if row == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#e0e0e0')

def evaluate_model(model, test_ar, device, output_folder, args):
    """Evaluate model on a test AR and generate plots."""
    model.eval()
    
    # Force rid_of_top = 1 for all test ARs
    rid_of_top = 1
    
    # Get AR-specific settings
    if test_ar == 11698:
        starting_tile = 46 - rid_of_top * SIZE
        before_plot = 50
        num_in = 96
        NOAA_first = datetime(2013, 3, 15)
        NOAA_second = datetime(2013, 3, 17)
    elif test_ar == 11726:
        starting_tile = 37 - rid_of_top * SIZE
        before_plot = 50
        num_in = 72
        NOAA_first = datetime(2013, 4, 20)
        NOAA_second = datetime(2013, 4, 22)
    elif test_ar == 13165:
        starting_tile = 28 - rid_of_top * SIZE
        before_plot = 40
        num_in = 96
        NOAA_first = datetime(2022, 12, 12)
        NOAA_second = datetime(2022, 12, 14)
    elif test_ar == 13179:
        starting_tile = 37 - rid_of_top * SIZE
        before_plot = 40
        num_in = 96
        NOAA_first = datetime(2022, 12, 30)
        NOAA_second = datetime(2023, 1, 1)
    elif test_ar == 13183:
        starting_tile = 37 - rid_of_top * SIZE
        before_plot = 40
        num_in = 96
        NOAA_first = datetime(2023, 1, 6)
        NOAA_second = datetime(2023, 1, 8)
    else:
        raise ValueError("Invalid test_AR value")
    
    NOAA1 = mdates.date2num(NOAA_first)
    NOAA2 = mdates.date2num(NOAA_second)
    
    # Load test data from .npz files
    power = np.load(f'/mmfs1/project/mx6/jst26/SAR_EMERGENCE_RESEARCH/data/AR{test_ar}/mean_pmdop{test_ar}_flat.npz', allow_pickle=True)
    cont = np.load(f'/mmfs1/project/mx6/jst26/SAR_EMERGENCE_RESEARCH/data/AR{test_ar}/mean_int{test_ar}_flat.npz', allow_pickle=True)
    mag = np.load(f'/mmfs1/project/mx6/jst26/SAR_EMERGENCE_RESEARCH/data/AR{test_ar}/mean_mag{test_ar}_flat.npz', allow_pickle=True)
    
    # Extract arrays from .npz files
    pm23, pm34, pm45, pm56 = power['arr_0'], power['arr_1'], power['arr_2'], power['arr_3']
    time_arr = power['arr_4']
    ii = cont['arr_0']
    mf = mag['arr_0']
    
    # Remove top and bottom rows using rid_of_top = 1
    sl = slice(rid_of_top*SIZE, -rid_of_top*SIZE)
    pm23, pm34, pm45, pm56 = pm23[sl,:], pm34[sl,:], pm45[sl,:], pm56[sl,:]
    ii = ii[sl,:]
    mf = mf[sl,:]
    
    # Stack power maps and normalize
    stacked = np.stack([pm23, pm34, pm45, pm56], axis=1)
    mp, Mp = stacked.min(), stacked.max()
    stacked = (stacked - mp) / (Mp - mp)
    
    # Normalize intensities and magnetic flux
    mi, Mi = ii.min(), ii.max()
    mm, Mm = mf.min(), mf.max()
    ii = (ii - mi) / (Mi - mi)
    mf = (mf - mm) / (Mm - mm)
    
    # Prepare inputs with magnetic flux as fifth channel
    inputs = np.concatenate([stacked, np.expand_dims(mf, 1)], axis=1)
    
    # Create figure for evaluation plots
    fig = plt.figure(figsize=(16, 46))
    fig.subplots_adjust(left=0.15, right=0.85, top=0.97, bottom=0.1)
    gs0 = gridspec.GridSpec(7, 1, figure=fig, hspace=.2)
    fut = NUM_PRED - 1
    thr = -0.01
    st = 4
    
    # Collect metrics for all tiles
    all_metrics = []
    
    for i in range(7):
        tile_idx = starting_tile + i
        disp = tile_idx + 10
        print(f"Tile {disp}")
        
        X_test, y_test = lstm_ready(tile_idx, inputs.shape[1], inputs, ii, num_in, NUM_PRED)
        X_test = X_test.to(device)
        
        # Update model's sequence length for this AR
        model.seq_len = num_in
        model.positional_encoding = model._generate_positional_encoding(num_in, args.hidden_size)
        
        with torch.no_grad():
            pred = model(X_test)[:, fut].cpu().numpy()
        
        true = y_test[:, fut].numpy()
        
        # Recalibrate predictions
        last = ii.shape[1] - true.shape[0] - 1
        pred = recalibrate(pred, ii[tile_idx, last])
        
        # Calculate metrics for this tile
        metrics = calculate_short_long_metrics(true, pred)
        all_metrics.append(metrics)
        
        # Get data for plotting
        before = ii[tile_idx, last-before_plot:last]
        tcut = time_arr[last-before_plot:last+true.shape[0]]
        tnum = mdates.date2num(tcut)
        nanarr = np.full(before.shape, np.nan)
        
        # Calculate derivatives
        d_true = np.gradient(smooth_with_numpy(np.concatenate((before, true))))
        d_pred = np.gradient(pred)
        
        # Create NaN padding for derivatives
        nan_pad = np.full(before_plot, np.nan)
        d_true_full = np.concatenate([nan_pad, d_true])
        d_pred_full = np.concatenate([nan_pad, d_pred])
        
        # Ensure all arrays have the same length
        min_len = min(len(tnum), len(d_true_full), len(d_pred_full))
        tnum = tnum[:min_len]
        d_true_full = d_true_full[:min_len]
        d_pred_full = d_pred_full[:min_len]
        
        # Create subplots
        gs1 = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs0[i], height_ratios=[18, 4, 4, 4], hspace=0.3)
        
        # Main intensity plot
        ax0 = fig.add_subplot(gs1[0])
        ax0.plot(tnum, np.concatenate((before, true)), 'k-', label='Observed Intensity')
        ax0.plot(tnum, np.concatenate((nanarr, pred)), 'r-', label='Transformer Prediction')
        ax0.axvline(NOAA1, color='magenta', linestyle='--', label='NOAA First Record')
        ax0.axvline(NOAA2, color='darkmagenta', linestyle='--', label='NOAA Second Record')
        ax0.set_title(f'Tile {disp}', fontsize=12)
        ax0.set_ylabel('Normalized Intensity', fontsize=9, labelpad=20)
        ax0.set_ylim([-0.1, 1.1])
        ax0.grid(True)
        ax0.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax0.tick_params(labelbottom=False)
        legend = ax0.legend(bbox_to_anchor=(1.033, .83, 0.223, 0.11), loc='upper left', borderaxespad=0, fontsize=10, framealpha=1, mode='expand')
        legend.get_frame().set_boxstyle('square', pad=1)
        
        # Add metrics table
        create_metrics_table(ax0, metrics)
        
        # Observed derivative plot
        ax1 = fig.add_subplot(gs1[1], sharex=ax0)
        # dObs/dt: plot full black, overlay limegreen for emergence
        ax1.plot(tnum, d_true_full, color='black', linewidth=1)
        ind_o = emergence_indication(d_true_full, thr, st)
        for j in range(len(d_true_full)-1):
            if ind_o[j] != 0:
                ax1.plot(tnum[j:j+2], d_true_full[j:j+2], color='limegreen', linewidth=1)
        ax1.set_ylabel('dObs/dt', fontsize=7, labelpad=10)
        ax1.set_ylim([-0.05, 0.05])
        ax1.set_yticks([0])
        ax1.grid(True)
        ax1.tick_params(labelbottom=False)
        
        # Predicted derivative plot
        ax2 = fig.add_subplot(gs1[2], sharex=ax0)
        # dPred/dt: plot full red, overlay limegreen for emergence
        ax2.plot(tnum, d_pred_full, color='red', linewidth=1)
        ind_p = emergence_indication(d_pred_full, thr, st)
        for j in range(len(d_pred_full)-1):
            if ind_p[j] != 0:
                ax2.plot(tnum[j:j+2], d_pred_full[j:j+2], color='limegreen', linewidth=1)
        ax2.set_ylabel('dPred/dt', fontsize=7, labelpad=10)
        ax2.set_ylim([-0.05, 0.05])
        ax2.set_yticks([0])
        ax2.grid(True)
        ax2.tick_params(labelbottom=False)
        
        # Error plot with date labels
        ax3 = fig.add_subplot(gs1[3], sharex=ax0)
        ax3.plot(tnum[before_plot:], np.abs(true - pred), 'r-')
        ax3.set_ylabel('|Error|', fontsize=8)
        ax3.grid(True)
        ax3.xaxis.set_major_locator(mdates.DayLocator())
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax3.set_xlabel('Date', fontsize=10)
        ax3.tick_params(labelbottom=True)
        ax3.set_xlim(tnum[0], tnum[-1])
    
    plt.tight_layout(rect=[0, 0, 0.8, 0.96])
    plt.subplots_adjust(right=0.8)
    plt.suptitle(f'Model Evaluation for AR {test_ar}', y=0.99)
    plt.figtext(0.5, 0.983, 
                f'Epochs: {EPOCHS} | Learning Rate: {args.learning_rate:.3f} | Hidden Size: {args.hidden_size} | '
                f'Layers: {args.num_layers} | Heads: {args.num_heads} | FF Ratio: {args.ff_ratio:.2f} | '
                f'Dropout: {args.dropout:.2f} | Time Window: {NUM_PRED} | Input Length: {NUM_IN} | '
                f'Rid of Top: {RID_OF_TOP} | Size: {SIZE} | Tiles: {TILES}',
                ha='center', fontsize=10)
    
    # Add summary table at the bottom
    table_ax = fig.add_axes([0.15, -0.045, 0.65, 0.12])
    table_ax.axis('off')
    
    # Add title for the summary table
    table_ax.text(0.5, 1, 'Model Parameters and Overall Performance Metrics', 
                 ha='center', va='center', fontsize=12)
    
    # Calculate mean metrics across all tiles
    def mean_metric(metrics_list, idx):
        vals = []
        for m in metrics_list:
            short_metrics, long_metrics = m
            if idx < len(short_metrics):
                vals.append(short_metrics[idx])
            else:
                vals.append(long_metrics[idx - len(short_metrics)])
        return np.mean(vals) if vals else 0.0
    
    # Prepare parameter rows
    param_headers = ["Parameter", "Value"]
    param_rows = [
        ["Time Window", str(NUM_PRED)],
        ["Rid of Top", str(rid_of_top)],  # Use rid_of_top = 1 in the table
        ["Input Len", str(num_in)],  # Use AR-specific input length
        ["Layers", str(args.num_layers)],
        ["Hidden", str(args.hidden_size)],
        ["Epochs", str(EPOCHS)],
        ["LR", f"{args.learning_rate:.3f}"],
    ]
    
    # Prepare metric rows
    metric_names = ['MSE', 'RMSE', 'MAE', 'R²']
    short_metrics = [
        [name, f"{mean_metric(all_metrics, i):.4f}"]
        for i, name in enumerate(metric_names)
    ]
    long_metrics = [
        [name, f"{mean_metric(all_metrics, i + len(metric_names)):.4f}"]
        for i, name in enumerate(metric_names)
    ]
    
    # Combine parameter rows and metric rows
    table_data = param_rows \
               + [['', '']] \
               + [['Metric', 'Value']] \
               + [['Short-term (first 50)', '']] \
               + short_metrics \
               + [['Long-term (remaining)', '']] \
               + long_metrics
    
    # Create table
    summary_table = table_ax.table(
        cellText=table_data,
        colLabels=param_headers,
        colColours=['#e0e0e0'] * len(param_headers),
        cellLoc='center',
        loc='upper center'
    )
    
    # Styling
    summary_table.auto_set_font_size(False)
    summary_table.set_fontsize(10)
    summary_table.scale(1, 1.3)
    
    # Add grid lines, bold header, and alternating row shading
    for (row, col), cell in summary_table.get_celld().items():
        cell.set_edgecolor('gray')
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_text_props(weight='bold')
        elif row % 2 == 1:
            cell.set_facecolor('#f9f9f9')
        else:
            cell.set_facecolor('white')
    
    # Save plot
    plt.savefig(os.path.join(output_folder, f'eval_{test_ar}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Return metrics without saving individual files
    return {
        'test_ar': test_ar,
        'short_term': {
            'mse': mean_metric(all_metrics, 0),
            'rmse': mean_metric(all_metrics, 1),
            'mae': mean_metric(all_metrics, 2),
            'r2': mean_metric(all_metrics, 3)
        },
        'long_term': {
            'mse': mean_metric(all_metrics, 4),
            'rmse': mean_metric(all_metrics, 5),
            'mae': mean_metric(all_metrics, 6),
            'r2': mean_metric(all_metrics, 7)
        }
    }

def main():
    args = parse_args()
    output_folder = get_output_folder(args)
    os.makedirs(output_folder, exist_ok=True)
    print(f'\nOutput will be saved to: {output_folder}')
    
    # Save hyperparameters including fixed parameters
    hyperparameters = {
        **vars(args),  # Command line arguments
        'num_pred': NUM_PRED,
        'rid_of_top': RID_OF_TOP,
        'num_in': NUM_IN,
        'epochs': EPOCHS,
        'size': SIZE,
        'tiles': TILES,
        'train_ars': TRAIN_ARs,
        'test_ars': TEST_ARs
    }
    
    # Save hyperparameters to JSON
    with open(os.path.join(output_folder, 'hyperparameters.json'), 'w') as f:
        json.dump(hyperparameters, f, indent=4)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Initialize model with max sequence length
    model = SpatioTemporalTransformer(
        input_dim=5,  # 4 power maps + 1 magnetic flux
        seq_len=110,  # Use max sequence length for training
        embed_dim=args.hidden_size,
        num_heads=args.num_heads,
        ff_dim=int(args.hidden_size * args.ff_ratio),
        num_layers=args.num_layers,
        output_dim=NUM_PRED,
        dropout=args.dropout
    ).to(device)
    
    # Load all ARs upfront
    print('Loading data and splitting into tiles for {} ARs'.format(len(TRAIN_ARs)))
    all_inputs = []
    all_intensities = []
    
    for ar in TRAIN_ARs:
        power = np.load(f'/mmfs1/project/mx6/jst26/SAR_EMERGENCE_RESEARCH/data/AR{ar}/mean_pmdop{ar}_flat.npz', allow_pickle=True)
        cont = np.load(f'/mmfs1/project/mx6/jst26/SAR_EMERGENCE_RESEARCH/data/AR{ar}/mean_int{ar}_flat.npz', allow_pickle=True)
        mag = np.load(f'/mmfs1/project/mx6/jst26/SAR_EMERGENCE_RESEARCH/data/AR{ar}/mean_mag{ar}_flat.npz', allow_pickle=True)
        
        # Extract arrays from .npz files
        pm23, pm34, pm45, pm56 = power['arr_0'], power['arr_1'], power['arr_2'], power['arr_3']
        ii = cont['arr_0']
        mf = mag['arr_0']
        
        # Remove top and bottom rows
        sl = slice(RID_OF_TOP*SIZE, -RID_OF_TOP*SIZE)
        pm23, pm34, pm45, pm56 = pm23[sl,:], pm34[sl,:], pm45[sl,:], pm56[sl,:]
        ii = ii[sl,:]
        mf = mf[sl,:]
        
        # Stack power maps and normalize
        stacked = np.stack([pm23, pm34, pm45, pm56], axis=1)
        mp, Mp = stacked.min(), stacked.max()
        stacked = min_max_scaling(stacked, mp, Mp)
        
        # Normalize intensities and magnetic flux
        mi, Mi = ii.min(), ii.max()
        mm, Mm = mf.min(), mf.max()
        ii = min_max_scaling(ii, mi, Mi)
        mf = min_max_scaling(mf, mm, Mm)
        
        # Prepare inputs with magnetic flux as fifth channel
        inputs = np.concatenate([stacked, np.expand_dims(mf, 1)], axis=1)
        
        # Append all ARs
        all_inputs.append(inputs)
        all_intensities.append(ii)
    
    # Stack all ARs
    all_inputs = np.stack(all_inputs, axis=-1)
    all_intensities = np.stack(all_intensities, axis=-1)
    input_size = np.shape(all_inputs)[1]
    
    # Train on each AR separately
    for ar_idx, ar in enumerate(TRAIN_ARs):
        print(f'\nTraining on AR {ar} ({ar_idx + 1} of {len(TRAIN_ARs)})...')
        
        # Get data for this AR
        power_maps = all_inputs[:,:,:,ar_idx]
        intensities = all_intensities[:,:,ar_idx]
        
        # Prepare data for this AR
        train_data = []
        train_targets = []
        
        # Use all tiles except first for training
        for tile in range(1, TILES):
            X, y = lstm_ready(tile, input_size, power_maps, intensities, NUM_IN, NUM_PRED)
            train_data.append(X)
            train_targets.append(y)
        
        # Use first tile for validation
        val_X, val_y = lstm_ready(0, input_size, power_maps, intensities, NUM_IN, NUM_PRED)
        
        train_data = torch.cat(train_data, dim=0)
        train_targets = torch.cat(train_targets, dim=0)
        
        # Create data loaders for this AR
        train_dataset = TensorDataset(train_data, train_targets)
        val_dataset = TensorDataset(val_X, val_y)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)
        
        # Initialize optimizer for this AR
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        
        # Train on this AR
        train_losses, val_losses = training_loop_w_stats(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            num_epochs=EPOCHS,
            device=device,
            output_folder=output_folder,
            warmup_ratio=args.warmup_ratio,
            args=args
        )
    
    # Evaluate on test ARs
    print('\nStarting evaluation...')
    all_metrics = []
    for test_idx, test_ar in enumerate(TEST_ARs):
        print(f'Evaluating on {test_ar} ({test_idx + 1} of {len(TEST_ARs)})...')
        metrics = evaluate_model(model, test_ar, device, output_folder, args)
        all_metrics.append(metrics)
    
    # Save all metrics to a single file
    metrics_path = os.path.join(output_folder, 'all_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=4)
    print(f'\nMetrics saved to: {metrics_path}')
    
    print('Training and evaluation complete!')

def plot_training_diagnostics(train_losses, val_losses, lr_history, output_folder, args):
    """Create comprehensive training diagnostics PDF with all plots on one page."""
    with PdfPages(os.path.join(output_folder, 'training_diagnostics.pdf')) as pdf:
        # Create a single figure with subplots
        fig = plt.figure(figsize=(15, 12))
        gs = gridspec.GridSpec(3, 2, figure=fig)
        
        # 1. Loss Curves
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(train_losses, label='Training Loss')
        ax1.plot(val_losses, label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 2. Learning Rate Schedule
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(lr_history, color='green', label='Learning Rate')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.legend()
        ax2.grid(True)
        
        # 3. Metrics Table
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.axis('off')
        
        # Calculate metrics
        train_mean = np.mean(train_losses)
        train_std = np.std(train_losses)
        val_mean = np.mean(val_losses)
        val_std = np.std(val_losses)
        best_epoch = np.argmin(val_losses)
        best_val_loss = min(val_losses)
        
        # Create metrics table
        metrics_data = [
            ['Metric', 'Value'],
            ['Training Loss Mean', f'{train_mean:.4f}'],
            ['Training Loss Std', f'{train_std:.4f}'],
            ['Validation Loss Mean', f'{val_mean:.4f}'],
            ['Validation Loss Std', f'{val_std:.4f}'],
            ['Best Epoch', str(best_epoch)],
            ['Best Validation Loss', f'{best_val_loss:.4f}'],
            ['Final Training Loss', f'{train_losses[-1]:.4f}'],
            ['Final Validation Loss', f'{val_losses[-1]:.4f}']
        ]
        
        table = ax3.table(
            cellText=metrics_data,
            loc='center',
            cellLoc='center',
            colWidths=[0.4, 0.4]
        )
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        for (row, col), cell in table.get_celld().items():
            cell.set_text_props(color='black')
            cell.set_facecolor('white')
            cell.set_edgecolor('#CCCCCC')
            cell.set_linewidth(0.5)
            if row == 0:
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#e0e0e0')
        
        # 4. Parameters Table
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')
        
        # Create parameters table with all parameters
        param_data = [
            ['Parameter', 'Value'],
            ['Time Window', str(NUM_PRED)],
            ['Rid of Top', str(RID_OF_TOP)],
            ['Input Length', str(NUM_IN)],
            ['Number of Layers', str(args.num_layers)],
            ['Hidden Size', str(args.hidden_size)],
            ['Number of Heads', str(args.num_heads)],
            ['FF Ratio', f'{args.ff_ratio:.2f}'],
            ['Dropout', f'{args.dropout:.2f}'],
            ['Learning Rate', f'{args.learning_rate:.3f}'],
            ['Warmup Ratio', f'{args.warmup_ratio:.2f}'],
            ['Epochs', str(EPOCHS)],
            ['Size', str(SIZE)],
            ['Tiles', str(TILES)],
            ['Train ARs', str(len(TRAIN_ARs))],
            ['Test ARs', str(len(TEST_ARs))]
        ]
        
        param_table = ax4.table(
            cellText=param_data,
            loc='center',
            cellLoc='center',
            colWidths=[0.4, 0.4]
        )
        
        # Style the parameters table
        param_table.auto_set_font_size(False)
        param_table.set_fontsize(10)
        param_table.scale(1.2, 1.5)
        
        for (row, col), cell in param_table.get_celld().items():
            cell.set_text_props(color='black')
            cell.set_facecolor('white')
            cell.set_edgecolor('#CCCCCC')
            cell.set_linewidth(0.5)
            if row == 0:
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#e0e0e0')
        
        plt.suptitle('Training Diagnostics Summary', fontsize=14, y=0.98)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

def training_loop_w_stats(model, train_loader, val_loader, optimizer, num_epochs, device, output_folder, warmup_ratio=0.1, args=None):
    """Training loop with warmup and cosine scheduling, reporting train/test losses."""
    train_losses = []
    val_losses = []
    lr_history = []
    best_val_loss = float('inf')
    
    # Calculate warmup epochs
    warmup_epochs = int(num_epochs * warmup_ratio)
    
    # Create warmup scheduler using LambdaLR
    def warmup_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch) / float(max(1, warmup_epochs))
        return 1.0
    
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lambda)
    
    # Create cosine scheduler for remaining steps
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs - warmup_epochs,
        eta_min=0.0
    )
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.mse_loss(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Update learning rate once per epoch
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()
        
        # Record learning rate
        lr_history.append(optimizer.param_groups[0]['lr'])
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += F.mse_loss(output, target).item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        print(f'Epoch: {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(output_folder, 'model.pth'))
    
    # Create comprehensive training diagnostics
    plot_training_diagnostics(train_losses, val_losses, lr_history, output_folder, args)
    
    return train_losses, val_losses

if __name__ == '__main__':
    main() 