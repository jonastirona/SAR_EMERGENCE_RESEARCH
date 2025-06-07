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
import glob
import re
from transformer.functions import (
    split_image, get_piece_means, dtws, lstm_ready, min_max_scaling,
    calculate_metrics, emergence_indication, smooth_with_numpy,
    recalibrate, add_grid_lines, highlight_tile, LSTM, calculate_extended_metrics
)
from transformer.models.st_transformer import SpatioTemporalTransformer

def get_ar_settings(test_AR, rid_of_top):
    """Get AR-specific settings."""
    if test_AR == 11698:
        starting_tile = 46 - rid_of_top * 9
        before_plot = 50
        num_in = 96
        NOAA_first = datetime(2013, 3, 15, 0, 0, 0)
        NOAA_second = datetime(2013, 3, 17, 0, 0, 0)
    elif test_AR == 11726:
        starting_tile = 37 - rid_of_top * 9
        before_plot = 50
        num_in = 72
        NOAA_first = datetime(2013, 4, 20, 0, 0, 0)
        NOAA_second = datetime(2013, 4, 22, 0, 0, 0)
    elif test_AR == 13165:
        starting_tile = 28 - rid_of_top * 9
        before_plot = 40
        num_in = 96
        NOAA_first = datetime(2022, 12, 12, 0, 0, 0)
        NOAA_second = datetime(2022, 12, 14, 0, 0, 0)
    elif test_AR == 13179:
        starting_tile = 37 - rid_of_top * 9
        before_plot = 40
        num_in = 96
        NOAA_first = datetime(2022, 12, 30, 0, 0, 0)
        NOAA_second = datetime(2023, 1, 1, 0, 0, 0)
    elif test_AR == 13183:
        starting_tile = 37 - rid_of_top * 9
        before_plot = 40
        num_in = 96
        NOAA_first = datetime(2023, 1, 6, 0, 0, 0)
        NOAA_second = datetime(2023, 1, 8, 0, 0, 0)
    else:
        raise ValueError("Invalid test_AR value")
    return starting_tile, before_plot, num_in, NOAA_first, NOAA_second

def evaluate_models_for_ar(test_AR, lstm_path, transformer_path):
    """Evaluate both LSTM and Transformer models for a given AR."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Evaluating AR {test_AR} on: {device} / Using {torch.cuda.device_count()} GPUs!')

    # Extract model parameters from transformer path
    matches = re.findall(r't(\d+)_r(\d+)_i(\d+)_n(\d+)_h(\d+)_e(\d+)_l([0-9.]+)\.pth', transformer_path)
    num_pred, rid_of_top, _, num_layers, hidden_size, n_epochs, learning_rate = (
        int(x) if i != 6 else float(x) for i, x in enumerate(matches[0])
    )
    
    # Get AR settings
    starting_tile, before_plot, num_in, NOAA_first, NOAA_second = get_ar_settings(test_AR, rid_of_top)
    NOAA_first_record = mdates.date2num(NOAA_first)
    NOAA_second_record = mdates.date2num(NOAA_second)

    # Load and preprocess data
    size = 9
    power_maps = np.load(f'/mmfs1/project/mx6/jst26/SAR_EMERGENCE_RESEARCH/data/AR{test_AR}/mean_pmdop{test_AR}_flat.npz', allow_pickle=True)
    mag_flux = np.load(f'/mmfs1/project/mx6/jst26/SAR_EMERGENCE_RESEARCH/data/AR{test_AR}/mean_mag{test_AR}_flat.npz', allow_pickle=True)
    intensities = np.load(f'/mmfs1/project/mx6/jst26/SAR_EMERGENCE_RESEARCH/data/AR{test_AR}/mean_int{test_AR}_flat.npz', allow_pickle=True)

    power_maps23 = power_maps['arr_0'][rid_of_top*size:-rid_of_top*size, :]
    power_maps34 = power_maps['arr_1'][rid_of_top*size:-rid_of_top*size, :]
    power_maps45 = power_maps['arr_2'][rid_of_top*size:-rid_of_top*size, :]
    power_maps56 = power_maps['arr_3'][rid_of_top*size:-rid_of_top*size, :]
    time = power_maps['arr_4']
    mag_flux = mag_flux['arr_0'][rid_of_top*size:-rid_of_top*size, :]
    intensities = intensities['arr_0'][rid_of_top*size:-rid_of_top*size, :]

    mag_flux[np.isnan(mag_flux)] = 0
    intensities[np.isnan(intensities)] = 0
    stacked_maps = np.stack([power_maps23, power_maps34, power_maps45, power_maps56], axis=1)
    stacked_maps[np.isnan(stacked_maps)] = 0
    
    # Normalize data
    min_p, max_p = np.min(stacked_maps), np.max(stacked_maps)
    min_m, max_m = np.min(mag_flux), np.max(mag_flux)
    min_i, max_i = np.min(intensities), np.max(intensities)
    stacked_maps = min_max_scaling(stacked_maps, min_p, max_p)
    mag_flux = min_max_scaling(mag_flux, min_m, max_m)
    intensities = min_max_scaling(intensities, min_i, max_i)
    mag_flux = np.expand_dims(mag_flux, axis=1)
    inputs = np.concatenate([stacked_maps, mag_flux], axis=1)

    # Initialize models
    input_size = 5  # Fixed input size (4 power maps + 1 magnetic flux)
    
    # Print debug information
    print(f"\nTransformer parameters:")
    print(f"hidden_size (embed_dim): {hidden_size}")
    print(f"num_heads: 10")
    print(f"num_layers: {num_layers}")
    print(f"num_pred: {num_pred}")
    print(f"input_size: {input_size}")
    print(f"num_in: {num_in}")
    
    # Initialize LSTM with the correct architecture from the trained model
    hidden_size = 64  # Original hidden size from checkpoint
    num_layers = 3  # Original number of layers from checkpoint
    
    lstm_model = LSTM(input_size, hidden_size, num_layers, num_pred).to(device)
    
    # Load LSTM weights
    lstm_state_dict = torch.load(lstm_path, map_location=device)
    new_lstm_state_dict = OrderedDict()
    for k, v in lstm_state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_lstm_state_dict[name] = v
    lstm_model.load_state_dict(new_lstm_state_dict)
    lstm_model.eval()

    # Initialize Transformer with the correct architecture
    transformer_model = SpatioTemporalTransformer(
        input_dim=input_size,
        seq_len=num_in,
        embed_dim=220,
        num_heads=10,
        ff_dim=int(220 * 3.41),
        num_layers=2,
        output_dim=12,
        dropout=0.16292645173701356
    ).to(device)
    
    # Load Transformer weights
    transformer_state_dict = torch.load(transformer_path, map_location=device)
    transformer_model.load_state_dict(transformer_state_dict)
    transformer_model.eval()

    # Setup plotting
    fig = plt.figure(figsize=(16, 20))
    main_gs = gridspec.GridSpec(7, 1, figure=fig, hspace=0.4)
    all_metrics_lstm = []
    all_metrics_transformer = []
    future = num_pred - 1
    threshold = -0.01
    sust_time = 4

    # Evaluate each tile
    for i in range(7):
        display_tile = starting_tile + 10 + i
        print(f"\nTile {display_tile}")
        
        # Prepare data - fix tile indexing
        # Map the display tile number back to 0-8 range for lstm_ready
        relative_tile = i  # This will be 0-6 for the 7 tiles we're looking at
        X_test, y_test = lstm_ready(relative_tile, size, inputs, intensities, num_in, num_pred)
        X_test = X_test.to(device)
        X_test_transformer = torch.reshape(X_test, (X_test.shape[0], num_in, X_test.shape[2]))
        
        # Get predictions
        with torch.no_grad():
            lstm_pred = lstm_model(X_test)[:, future].cpu().numpy()
            transformer_pred = transformer_model(X_test_transformer)[:, future].cpu().numpy()
        true = y_test[:, future].numpy()
        
        # Recalibrate predictions
        last_known_idx = intensities.shape[1] - true.shape[0] - 1
        lstm_pred = recalibrate(lstm_pred, intensities[relative_tile, last_known_idx])
        transformer_pred = recalibrate(transformer_pred, intensities[relative_tile, last_known_idx])

        # Setup subplots
        gs = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=main_gs[i], height_ratios=[4, 1, 1], hspace=0.05)
        int_before_pred = intensities[relative_tile, last_known_idx-before_plot:last_known_idx]
        time_cut = time[last_known_idx-before_plot:last_known_idx+np.shape(true)[0]]
        time_cut_mpl = mdates.date2num(time_cut)
        nan_array = np.full(int_before_pred.shape, np.nan)

        # Main prediction plot
        ax0 = plt.subplot(gs[0])
        # Plot raw data and predictions with full opacity
        ax0.plot(time_cut_mpl, np.concatenate((int_before_pred, true)), 'k-', label='Observed', alpha=0.7)
        ax0.plot(time_cut_mpl, np.concatenate((nan_array, lstm_pred)), 'b-', label='LSTM Prediction', alpha=0.7)
        ax0.plot(time_cut_mpl, np.concatenate((nan_array, transformer_pred)), 'r-', label='Transformer Prediction', alpha=0.7)
        # Plot smoothed data with lower opacity
        ax0.plot(time_cut_mpl, smooth_with_numpy(np.concatenate((int_before_pred, true))), 'k-', label='Observed (Smoothed)', alpha=0.25)
        ax0.axvline(x=NOAA_first_record, color='magenta', linestyle='--', label='NOAA 1st Record')
        ax0.axvline(x=NOAA_second_record, color='darkmagenta', linestyle='--', label='After Emergence')
        ax0.set_ylabel(f'Normalized Intensity\nTile {display_tile}')
        ax0.set_ylim([-0.1, 1.1])
        ax0.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5)
        ax0.legend(loc='upper right', fontsize=7)
        ax0.tick_params(axis='x', which='both', labelbottom=False)
        ax0.xaxis_date()
        ax0.xaxis.set_major_locator(mdates.DayLocator())
        ax0.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

        # Derivative plots
        ax1 = plt.subplot(gs[1])
        d_true = np.gradient(smooth_with_numpy(np.concatenate((int_before_pred, true))))
        d_lstm = np.gradient(lstm_pred)
        d_transformer = np.gradient(transformer_pred)
        
        # First derivative plot (Observed)
        indicator_true = emergence_indication(d_true, threshold, sust_time)
        first = True
        for j in range(len(d_true) - 1):
            current_color = 'g' if indicator_true[j] == 0 else 'r'
            if current_color == 'r' and first:
                readable_time = [mdates.num2date(time).strftime('%Y-%m-%d %H:%M:%S') for time in time_cut_mpl[j:j+2]]
                first = False
                print('Observed First Emergence Time: {}'.format(readable_time[1]))
            ax1.plot(time_cut_mpl[j:j+2], d_true[j:j+2], color=current_color)
        ax1.axvline(x=NOAA_first_record, color='magenta', linestyle='--')
        ax1.axvline(x=NOAA_second_record, color='darkmagenta', linestyle='--')
        ax1.set_ylim([-0.05, 0.05])
        ax1.set_yticks([0])
        ax1.grid(True)
        ax1.set_ylabel(r'$\frac{d Obs}{dt}$')
        ax1.set_xticklabels([])

        # Second derivative plot (Predictions)
        ax2 = plt.subplot(gs[2])
        # LSTM derivative
        d_lstm = np.concatenate((np.full(int_before_pred.shape, 0), d_lstm))
        indicator_lstm = emergence_indication(d_lstm[before_plot:], threshold, sust_time)
        indicator_lstm = np.concatenate((np.full(int_before_pred.shape, np.nan), indicator_lstm))
        first = True
        for k in range(len(d_lstm) - 1):
            current_color = 'b' if indicator_lstm[k] == 0 else 'r' if indicator_lstm[k] == 1 else 'grey'
            if current_color == 'r' and first:
                readable_time = [mdates.num2date(time).strftime('%Y-%m-%d %H:%M:%S') for time in time_cut_mpl[k:k+2]]
                first = False
                print('LSTM First Emergence Time: {}'.format(readable_time[1]))
            alph = 1 if indicator_lstm[k] in [0, 1] else 0
            ax2.plot(time_cut_mpl[k:k+2], d_lstm[k:k+2], color=current_color, alpha=alph)

        # Transformer derivative
        d_transformer = np.concatenate((np.full(int_before_pred.shape, 0), d_transformer))
        indicator_transformer = emergence_indication(d_transformer[before_plot:], threshold, sust_time)
        indicator_transformer = np.concatenate((np.full(int_before_pred.shape, np.nan), indicator_transformer))
        first = True
        for k in range(len(d_transformer) - 1):
            current_color = 'r' if indicator_transformer[k] == 0 else 'darkred' if indicator_transformer[k] == 1 else 'grey'
            if current_color == 'darkred' and first:
                readable_time = [mdates.num2date(time).strftime('%Y-%m-%d %H:%M:%S') for time in time_cut_mpl[k:k+2]]
                first = False
                print('Transformer First Emergence Time: {}'.format(readable_time[1]))
            alph = 1 if indicator_transformer[k] in [0, 1] else 0
            ax2.plot(time_cut_mpl[k:k+2], d_transformer[k:k+2], color=current_color, alpha=alph, linestyle='--')

        ax2.axvline(x=NOAA_first_record, color='magenta', linestyle='--')
        ax2.axvline(x=NOAA_second_record, color='darkmagenta', linestyle='--')
        ax2.xaxis_date()
        ax2.xaxis.set_major_locator(mdates.DayLocator())
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%y'))
        ax2.tick_params(axis='x', which='major', labelsize=9, rotation=45)
        ax2.set_ylim([-0.05, 0.05])
        ax2.set_yticks([0])
        ax2.grid(True)
        ax2.set_ylabel(r'$\frac{d Pred}{dt}$')

        # Calculate metrics
        metrics_lstm = calculate_extended_metrics(lstm_model, true, lstm_pred, training_time=0)  # Set training_time=0 since we're evaluating
        metrics_transformer = calculate_extended_metrics(transformer_model, true, transformer_pred, training_time=0)
        all_metrics_lstm.append(metrics_lstm)
        all_metrics_transformer.append(metrics_transformer)
        
        # Print metrics
        print(f"\nTile {display_tile} Metrics:")
        print(f"LSTM - MAE: {metrics_lstm['MAE']:.3f}, RMSE: {metrics_lstm['RMSE']:.3f}, R2: {metrics_lstm['R2']:.3f}")
        print(f"Transformer - MAE: {metrics_transformer['MAE']:.3f}, RMSE: {metrics_transformer['RMSE']:.3f}, R2: {metrics_transformer['R2']:.3f}")

    # Calculate mean metrics only if we have valid metrics
    if all_metrics_lstm and all_metrics_transformer:
        # Get a list of all metric names from the first metrics dictionary
        metric_names = list(all_metrics_lstm[0].keys())
        
        # Calculate mean metrics, handling potential None values
        mean_metrics_lstm = {}
        mean_metrics_transformer = {}
        for name in metric_names:
            lstm_values = [m[name] for m in all_metrics_lstm if m[name] is not None]
            transformer_values = [m[name] for m in all_metrics_transformer if m[name] is not None]
            
            mean_metrics_lstm[name] = np.mean(lstm_values) if lstm_values else None
            mean_metrics_transformer[name] = np.mean(transformer_values) if transformer_values else None

        # Create comparison table
        metrics_text = (
            'Model Comparison Metrics:\n\n'
            f"{'Metric':<15} {'LSTM':<12} {'Transformer':<12}\n"
            f"{'-'*40}\n"
        )
        
        for name in metric_names:
            lstm_value = mean_metrics_lstm[name]
            transformer_value = mean_metrics_transformer[name]
            
            if isinstance(lstm_value, (int, float)) and isinstance(transformer_value, (int, float)):
                if name == 'params':
                    metrics_text += f"{'Parameters':<15} {lstm_value/1e6:.1f}M {transformer_value/1e6:.1f}M\n"
                elif name == 'train_time':
                    metrics_text += f"{'Train Time':<15} {lstm_value:.1f}min {transformer_value:.1f}min\n"
                else:
                    metrics_text += f"{name:<15} {lstm_value:.3f} {transformer_value:.3f}\n"

    # Add metrics table to plot
    fig.text(0.5, 0.02, metrics_text, ha='center', va='bottom', fontsize=10, family='monospace')

    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.2)  # Increased bottom margin for metrics table
    plt.suptitle(f'Model Comparison for Active Region {test_AR}\n(Time Window={num_pred}h, Rid of Top={rid_of_top}, Input Length={num_in}h)', y=0.98)

    # Save results
    save_path = f"/mmfs1/project/mx6/jst26/SAR_EMERGENCE_RESEARCH/evaluation/results/AR{test_AR}_comparison.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Saved comparison plot at: {save_path}')
    plt.close()

if __name__ == "__main__":
    # Model paths
    lstm_path = "/mmfs1/project/mx6/jst26/SAR_EMERGENCE_RESEARCH/lstm/results/t12_r4_i110_n3_h64_e1000_l0.01.pth"
    transformer_path = "/mmfs1/project/mx6/jst26/SAR_EMERGENCE_RESEARCH/transformer/results/st_transformer/t12_r4_i110_n2_h220_e500_l0.004099697696351005.pth"
    
    # ARs to evaluate
    test_ARs = [11698, 11726, 13165, 13179, 13183]
    
    # Run evaluation for each AR
    for ar in test_ARs:
        evaluate_models_for_ar(ar, lstm_path, transformer_path) 