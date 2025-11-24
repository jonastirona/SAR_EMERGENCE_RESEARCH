# functions to be used in pipeline

import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
from astropy.io import fits
import sys
from PIL import Image
import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from scipy.signal import detrend
from scipy.signal import detrend
import warnings
import random
from ray import tune
import re
from collections import OrderedDict
import glob

isVanillaLSTM = True
if isVanillaLSTM:
    model_type = "VanillaLSTM"
else:
    model_type = "LSTM"
warnings.filterwarnings("ignore")
l = re.split(r"[\\/]", os.path.abspath(os.getcwd()))
BASE_PATH = "/".join(l[:-1]) + "/"

DATA_PATH = BASE_PATH + "SAR_EMERGENCE_RESEARCH/data"
RESULTS_PATH = BASE_PATH + "SAR_EMERGENCE_RESEARCH/lstm/results/"
MODELS_PATH = BASE_PATH + "SAR_EMERGENCE_RESEARCH/lstm/models"
THRESHOLD = 0.01
SUST_TIME = 4


##### Sept 18th and later
def min_max_scaling(arr, min_val, max_val):
    """
    Set values
    """
    return (arr - min_val) / (max_val - min_val)


def lstm_ready(
    tile, size, power_maps, mag_flux, num_in, num_pred
):  # ,min_p,max_p,min_i,max_i):
    # Read AR and create lstm ready data
    final_maps = np.transpose(power_maps, axes=(2, 1, 0))
    final_flux = np.transpose(mag_flux, axes=(1, 0))
    X_trans = final_maps[:, :, tile]
    y_trans = final_flux[:, tile]
    X_ss, y_mm, last_vals = split_sequences(X_trans, y_trans, num_in, num_pred)
    return torch.Tensor(X_ss), torch.Tensor(y_mm), torch.Tensor(last_vals)


class LSTM(nn.Module):
    # __init__ stays the same...
    def __init__(self, input_size, hidden_size, num_layers, output_length, dropout=0.0):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_length = output_length
        self.encoder_lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=dropout
        )
        self.decoder_lstm = nn.LSTM(
            1, hidden_size, num_layers, batch_first=True, dropout=dropout
        )

        self.decoder_fc = nn.Linear(hidden_size, 1)

    # The forward pass now accepts the target tensor 'y' for teacher forcing
    def forward(self, x, y=None, teacher_forcing_ratio=0.5):
        # Encoder
        _, (hidden, cell) = self.encoder_lstm(x)

        # Decoder
        decoder_input = torch.zeros(x.size(0), 1, 1).to(x.device)
        outputs = []

        for t in range(self.output_length):
            out_dec, (hidden, cell) = self.decoder_lstm(decoder_input, (hidden, cell))
            out = self.decoder_fc(out_dec)
            outputs.append(out)

            # Decide whether to use teacher forcing for the next step
            use_teacher_forcing = (y is not None) and (
                random.random() < teacher_forcing_ratio
            )

            if use_teacher_forcing:
                # Use the actual ground-truth value as the next input
                decoder_input = y[:, t].unsqueeze(1).unsqueeze(1)
            else:
                # Use the model's own prediction as the next input
                decoder_input = out

        outputs = torch.cat(outputs, dim=1).squeeze(-1)
        return outputs


class VanillaLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_length, dropout=0.0):
        super(VanillaLSTM, self).__init__()

        # A single LSTM layer that processes the entire input sequence
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # Input tensor shape: [batch_size, seq_length, input_size]
            dropout=dropout,
        )

        # A single linear layer to map the final LSTM state to the desired output length
        self.fc = nn.Linear(hidden_size, output_length)

    def forward(self, x):
        # Pass the input sequence through the LSTM layer
        # We don't need the final hidden and cell states, just the output sequence
        lstm_out, _ = self.lstm(x)

        # We only need the output from the very last time step of the sequence
        # lstm_out has shape [batch_size, seq_length, hidden_size]
        # We take the last time step: lstm_out[:, -1, :]
        last_time_step_out = lstm_out[:, -1, :]

        # Pass the last time step's output to the linear layer to get the final prediction
        prediction = self.fc(last_time_step_out)

        return prediction


# split a multivariate sequence past, future samples (X and y)
def split_sequences(input_sequences, output_sequences, n_steps_in, n_steps_out):
    X, y = list(), list()  # instantiate X and y
    last_vals = list()
    for i in range(len(input_sequences)):
        # find the end of the input, output sequence
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out - 1
        # check if we are beyond the dataset
        if out_end_ix > len(input_sequences):
            break
        # gather input and output of the pattern
        seq_x, seq_y = (
            input_sequences[i:end_ix],
            output_sequences[end_ix - 1 : out_end_ix],
        )
        last_val = output_sequences[
            end_ix - 2
        ]  # for calibration purposes when doing RMSE on validation
        X.append(seq_x), y.append(seq_y), last_vals.append(last_val)
    return np.array(X), np.array(y), np.array(last_vals)


def calculate_metrics(timeline_true, timeline_predicted):
    # Ensure inputs are NumPy arrays for consistency
    timeline_true = np.array(timeline_true)
    timeline_predicted = np.array(timeline_predicted)
    # Calculate Mean Absolute Error (MAE)
    MAE = np.mean(np.abs(timeline_predicted - timeline_true))
    # Calculate Mean Squared Error (MSE)
    MSE = np.mean(np.square(timeline_predicted - timeline_true))
    # Calculate Root Mean Squared Error (RMSE)
    RMSE = np.sqrt(MSE)
    # Calculate Root Mean Squared Logarithmic Error (RMSLE)
    RMSLE = np.sqrt(
        np.mean(np.square(np.log1p(timeline_predicted) - np.log1p(timeline_true)))
    )
    # Calculate R-squared (R²)
    SS_res = np.sum(np.square(timeline_true - timeline_predicted))
    SS_tot = np.sum(np.square(timeline_true - np.mean(timeline_true)))
    R_squared = 1 - (SS_res / SS_tot)
    return MAE, MSE, RMSE, RMSLE, R_squared


def emergence_indication(d_true, threshold, sust_time):
    d_true = smooth_with_numpy(d_true)
    indicator = np.zeros(d_true.shape)  # Initialize with 0s (green)
    # Populate the indicator array
    for j in range(len(d_true)):
        if d_true[j] >= threshold:
            indicator[j] = 1  # Mark as red
    # Enforce the sustained condition
    sustained = True
    if sustained:
        start_idx = None
        for i in range(len(indicator)):
            if indicator[i] == 1 and start_idx is None:
                start_idx = i  # Start of a red sequence
            elif (
                indicator[i] == 0 and start_idx is not None
            ):  # End of a red sequence, check its length
                if i - start_idx < sust_time:
                    indicator[start_idx:i] = 0  # Sequence too short, revert to green
                start_idx = None  # Reset start index for the next sequence
        # Check for a sequence that goes till the end of the array
        if start_idx is not None and len(indicator) - start_idx < sust_time:
            indicator[start_idx:] = 0
    return indicator


def smooth_with_numpy(d_true, window_size=5):
    if window_size <= 1:
        return d_true
    pad_width = window_size // 2  # Calculate the number of elements to pad on each side
    padded_d_true = np.pad(
        d_true, pad_width, mode="edge"
    )  # Pad the beginning and end of d_true with its first and last values, respectively
    window = np.ones(window_size) / window_size  # Create the smoothing window
    smoothed_d_true = np.convolve(
        padded_d_true, window, mode="same"
    )  # Apply convolution on the padded data
    return (
        smoothed_d_true[pad_width:-pad_width] if pad_width else smoothed_d_true
    )  # Remove the padding to return the smoothed array to its original length


def recalibrate(pred, previous_value):
    trend = pred - pred[0]
    new_pred = trend + previous_value
    return new_pred


def highlight_tile(ax, tile_number, divisions=9, color="r", linewidth=1):
    """
    Highlights a specific tile in the grid with a colored box.

    Parameters:
    - ax: The axes object on which the grid and image are plotted.
    - tile_number: The number of the tile to highlight, in row-major order.
    - divisions: The number of divisions along each axis (assumes a square grid).
    - color: Color of the highlight box.
    - linewidth: Width of the highlight box lines.
    """
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Calculate width and height of each tile
    tile_width = (xlim[1] - xlim[0]) / divisions
    tile_height = (ylim[1] - ylim[0]) / divisions

    # Calculate row and column index of the tile (0-indexed)
    row_idx = (tile_number - 1) // divisions
    col_idx = (tile_number - 1) % divisions

    # Calculate coordinates for the bottom-left corner of the tile
    x = xlim[0] + col_idx * tile_width
    y = ylim[1] - (row_idx + 1) * tile_height  # y coordinates go top-to-bottom

    # Create a rectangle patch to highlight the tile
    from matplotlib.patches import Rectangle

    rect = Rectangle(
        (x, y),
        tile_width,
        tile_height,
        linewidth=linewidth,
        edgecolor=color,
        facecolor="none",
    )
    ax.add_patch(rect)


def load_ar_data(ar_num, size, rid_of_top, starting_tile):
    """Loads and preprocesses data for a single Active Region (AR)."""
    try:
        # Load data from .npz files
        pm_path = os.path.join(DATA_PATH, f"AR{ar_num}", f"mean_pmdop{ar_num}_flat.npz")
        mag_path = os.path.join(DATA_PATH, f"AR{ar_num}", f"mean_mag{ar_num}_flat.npz")
        int_path = os.path.join(DATA_PATH, f"AR{ar_num}", f"mean_int{ar_num}_flat.npz")

        power_maps = np.load(pm_path, allow_pickle=True)
        mag_flux_data = np.load(mag_path, allow_pickle=True)
        intensities_data = np.load(int_path, allow_pickle=True)
        time = power_maps["arr_4"]

        # Unpack arrays
        power_maps = [power_maps[f"arr_{i}"] for i in range(4)]
        mag_flux = mag_flux_data["arr_0"]
        intensities = intensities_data["arr_0"]

        # Trim, stack, and handle NaNs
        trim_slice = slice(starting_tile, starting_tile + size)
        power_maps = [pm[trim_slice, :] for pm in power_maps]
        mag_flux = mag_flux[trim_slice, :]
        intensities = intensities[trim_slice, :]

        stacked_maps = np.stack(power_maps, axis=1)
        stacked_maps[np.isnan(stacked_maps)] = 0
        mag_flux[np.isnan(mag_flux)] = 0
        intensities[np.isnan(intensities)] = 0

        return stacked_maps, mag_flux, intensities, time

    except FileNotFoundError:
        print(f"Warning: Data files for AR {ar_num} not found. Skipping.")
        return None, None, None


def process_data(maps, flux, cont_int, m_scale, f_scale, cont_int_scale):
    stacked_maps = min_max_scaling(maps, *m_scale)
    mag_flux = min_max_scaling(flux, *f_scale)
    intensities = min_max_scaling(cont_int, *cont_int_scale)

    # Reshape int to have an extra dimension and then put it with pmaps
    int_reshaped = np.expand_dims(intensities, axis=1)

    inputs = np.concatenate([stacked_maps, int_reshaped], axis=1)

    return inputs, mag_flux


def get_params(path, file):
    matches = re.findall(
        r"pred(\d+)_r(\d+)_i(\d+)_n(\d+)_h(\d+)_e(\d+)_lr([0-9.]+)_d([0-9.]+)\.pth",
        file,
    )  # Extract numbers from the filename
    (
        num_pred,
        rid_of_top,
        num_in,
        num_layers,
        hidden_size,
        n_epochs,
        learning_rate,
        dropout,
    ) = [
        float(val) if i >= 6 else int(val) for i, val in enumerate(matches[0])
    ]  # Unpack the matched values into variables
    return (
        num_pred,
        rid_of_top,
        num_in,
        num_layers,
        hidden_size,
        n_epochs,
        learning_rate,
        dropout,
    )


def AR_defs(val_AR):
    before_plot, num_in, NOAA_first, NOAA_second = None, None, None, None
    end = 0
    start = 0
    if val_AR == 11698:
        window_s = 74
        starting_tile = 46
        before_plot = 50
        num_in = 96
        NOAA_first = datetime(2013, 3, 15, 0, 0, 0)
        NOAA_second = datetime(2013, 3, 17, 0, 0, 0)
    elif val_AR == 11726:
        window_s = 50
        starting_tile = 37
        before_plot = 50
        end = -24
        num_in = 72  # decrease even more -> 60
        NOAA_first = datetime(2013, 4, 20, 0, 0, 0)
        NOAA_second = datetime(2013, 4, 22, 0, 0, 0)
    elif val_AR == 13165:
        window_s = 40
        starting_tile = 28
        before_plot = 40
        num_in = 96
        end = -12
        NOAA_first = datetime(2022, 12, 12, 0, 0, 0)
        NOAA_second = datetime(2022, 12, 14, 0, 0, 0)
    elif val_AR == 13179:
        window_s = 40
        starting_tile = 37
        before_plot = 40
        num_in = 96
        end = -12
        NOAA_first = datetime(2022, 12, 30, 0, 0, 0)
        NOAA_second = datetime(2023, 1, 1, 0, 0, 0)
    elif val_AR == 13183:
        window_s = 40
        starting_tile = 37
        before_plot = 40
        num_in = 96
        end = -12
        NOAA_first = datetime(2023, 1, 6, 0, 0, 0)
        NOAA_second = datetime(2023, 1, 8, 0, 0, 0)
    else:
        print(
            "Invalid validation Active Region value. Please use 11698, 11726, 13165, 13179, or 13183."
        )
    return (
        before_plot,
        num_in,
        NOAA_first,
        NOAA_second,
        starting_tile,
        window_s,
        end,
        start,
    )


# --- Data Loading & Preparation ---
def prepare_dataset(
    ar_list,
    size,
    rid_of_top,
    num_in,
    num_pred,
    m_scale=None,
    flux_scale=None,
    cont_int_scale=None,
):
    """Builds a complete dataset (X, y) for a list of ARs."""
    all_inputs_list, all_flux_list = [], []
    all_maps = []
    all_flux = []
    all_cont_int = []

    # Load data for all ARs
    if m_scale is None:
        for ar in ar_list:
            maps, flux, cont_int, time = load_ar_data(
                ar, size, rid_of_top, size * rid_of_top
            )
            all_maps.append(maps)
            all_flux.append(flux)
            all_cont_int.append(cont_int)

        m_scale = (np.min(all_maps), np.max(all_maps))
        flux_scale = (np.min(all_flux), np.max(all_flux))
        cont_int_scale = (np.min(all_cont_int), np.max(all_cont_int))

        for i in range(len(all_maps)):
            combined_inputs, flux = process_data(
                all_maps[i],
                all_flux[i],
                all_cont_int[i],
                m_scale,
                flux_scale,
                cont_int_scale,
            )
            all_inputs_list.append(combined_inputs)
            all_flux_list.append(flux)
    else:
        for ar in ar_list:
            maps, flux, cont_int, time = load_ar_data(
                ar, size, rid_of_top, size * rid_of_top
            )
            combined_inputs, flux = process_data(
                maps, flux, cont_int, m_scale, flux_scale, cont_int_scale
            )
            all_inputs_list.append(combined_inputs)
            all_flux_list.append(flux)

    if not all_inputs_list:
        print("all_inputs_list does not exist")
        return None, None, 0

    # Create sequences for the LSTM
    x_list, y_list, last_list = [], [], []
    tiles = size**2 - 2 * size * rid_of_top

    for inputs, flux in zip(all_inputs_list, all_flux_list):
        for tile in range(tiles):
            x_seq, y_seq, last_seq = lstm_ready(
                tile, size, inputs, flux, num_in, num_pred
            )
            if x_seq.shape[0] > 0:
                x_seq = torch.reshape(x_seq, (x_seq.shape[0], num_in, x_seq.shape[2]))
                x_list.append(x_seq)
                y_list.append(y_seq)
                last_list.append(last_seq)

    if not x_list:
        print("X_list does not exist")
        return None, None, 0

    x_all = torch.cat(x_list, dim=0)
    y_all = torch.cat(y_list, dim=0)
    last_all = torch.cat(last_list, dim=0)  # Concatenate the last values
    input_feature_size = x_all.shape[2]

    return (
        x_all,
        y_all,
        last_all,
        input_feature_size,
        m_scale,
        flux_scale,
        cont_int_scale,
    )


# --- Model Training & Evaluation ---
def train_epochHybridLSTM(
    model, dataloader, loss_fn, optimizer, device, teacher_ratio, alpha
):
    model.train()
    total_loss = 0
    loss_scaler = 100.0

    # Weighting factor for the two loss components
    # alpha

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        outputs = model(x, y, teacher_forcing_ratio=teacher_ratio)

        # 1. Calculate loss on the actual values
        value_loss = loss_fn(outputs, y)

        # 2. Calculate loss on the derivatives
        outputs_deriv = outputs[:, 1:] - outputs[:, :-1]
        y_deriv = y[:, 1:] - y[:, :-1]
        derivative_loss = loss_fn(outputs_deriv, y_deriv)

        # 3. Combine them into a hybrid loss
        loss = alpha * value_loss + (1 - alpha) * derivative_loss

        scaled_loss = loss * loss_scaler
        scaled_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def train_epochTeacherForcingLSTM(
    model, dataloader, loss_fn, optimizer, device, teacher_forcing
):
    """Runs a single training epoch."""

    model.train()

    total_loss = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        outputs = model(x, y, teacher_forcing_ratio=teacher_forcing)

        loss = loss_fn(outputs, y)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def train_epochHybridVanillaLSTM(model, dataloader, loss_fn, optimizer, device, alpha):
    model.train()
    total_loss = 0
    loss_scaler = 100.0

    # Weighting factor for the two loss components
    # alpha

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        outputs = model(x)

        # 1. Calculate loss on the actual values
        value_loss = loss_fn(outputs, y)

        # 2. Calculate loss on the derivatives
        outputs_deriv = outputs[:, 1:] - outputs[:, :-1]
        y_deriv = y[:, 1:] - y[:, :-1]
        derivative_loss = loss_fn(outputs_deriv, y_deriv)

        # 3. Combine them into a hybrid loss
        loss = alpha * value_loss + (1 - alpha) * derivative_loss

        scaled_loss = loss * loss_scaler
        scaled_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def train_epoch(model, dataloader, loss_fn, optimizer, device):
    """Runs a single training epoch."""

    model.train()

    total_loss = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        outputs = model(x)

        loss = loss_fn(outputs, y)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def get_first_emergence_index(signal, threshold, sust_time):
    """
    Returns the INDEX where emergence starts.
    """
    # signal = np.gradient(smooth_with_numpy(signal))
    indicator = emergence_indication(signal, threshold, sust_time)
    emergence_indices = np.where(indicator == 1)[0]
    if len(emergence_indices) > 0:
        return emergence_indices[0]
    return None


def calculate_lead_time_score(
    y_true_batch, y_pred_batch, threshold, sust_time, horizon_hours=12.0
):
    """
    Calculates the Mean Lead Time (in hours).
    Goal: MAXIMIZE this metric.

    Args:
        horizon_hours: The real-world duration the sequence represents (e.g., 12 hours).
    """
    scores = []
    seq_len = y_true_batch.shape[1]

    # Calculate how many hours one array index represents
    hours_per_step = horizon_hours / seq_len

    # Define Penalties (in hours)
    # Penalty for missing an event that happened (False Negative)
    missed_event_penalty = -2.0
    # Penalty for predicting an event that never happened (False Positive)
    false_alarm_penalty = -2.0

    for i in range(len(y_true_batch)):
        true_idx = get_first_emergence_index(y_true_batch[i], threshold, sust_time)
        pred_idx = get_first_emergence_index(y_pred_batch[i], threshold, sust_time)

        # 1. Successful Detection (Hit)
        if true_idx is not None and pred_idx is not None:
            true_time_hours = true_idx * hours_per_step
            pred_time_hours = pred_idx * hours_per_step

            # Lead Time = True - Pred
            # If True=10h, Pred=4h -> Lead=6h (Positive/Good)
            # If True=4h,  Pred=6h -> Lead=-2h (Negative/Late)
            lead_time = true_time_hours - pred_time_hours
            scores.append(lead_time)

        # 2. False Negative (Event happened, Model missed it)
        elif true_idx is not None and pred_idx is None:
            # You get a negative score for failing to predict a real event
            scores.append(missed_event_penalty)

        # 3. False Positive (No event, Model predicted one)
        elif true_idx is None and pred_idx is not None:
            # You get a negative score for crying wolf
            scores.append(false_alarm_penalty)

        # 4. True Negative (Nothing happened, Model predicted nothing)
        elif true_idx is None and pred_idx is None:
            # Neutral score, or 0. Since we want to maximize lead time,
            # 0 is an acceptable "neutral" for non-events.
            scores.append(0.0)

    return np.mean(scores)


def calculate_horizon_deficit(
    y_true_batch, y_pred_batch, threshold, sust_time, horizon_hours=12.0
):
    """
    Calculates deficit based on GRADIENTS (Rate of Change).
    Uses np.gradient (central difference) to maintain sequence length.
    Goal: MINIMIZE this value (0 is perfect).
    """
    deficits = []

    # 1. Calculate Gradients along the time axis (axis=1)
    # np.gradient returns a list of arrays if not handled carefully,
    # but specifying axis=1 ensures we get the gradient for time.
    true_grads = np.gradient(y_true_batch, axis=1)
    pred_grads = np.gradient(y_pred_batch, axis=1)

    # Sequence length remains the same as input (unlike np.diff)
    seq_len = true_grads.shape[1]
    hours_per_step = horizon_hours / seq_len

    missed_penalty = horizon_hours
    false_alarm_penalty = horizon_hours

    # Debug counters
    n_hits = 0
    n_misses = 0
    n_false_alarms = 0
    n_true_negatives = 0

    for i in range(len(true_grads)):
        # Pass the GRADIENT vector to the emergence checker
        true_idx = get_first_emergence_index(true_grads[i], threshold, sust_time)
        pred_idx = get_first_emergence_index(pred_grads[i], threshold, sust_time)

        # 1. Hit (Event happened & predicted)
        if true_idx is not None and pred_idx is not None:
            true_time = true_idx * hours_per_step
            pred_time = pred_idx * hours_per_step

            # Lead Time = True - Pred
            raw_lead = true_time - pred_time

            lead_time = min(max(raw_lead, 0.0), horizon_hours)
            # Deficit = Distance from ideal 12h lead
            # 12 - (12) = 0 (Perfect)
            # 12 - (-2) = 14 (Late)
            deficit = horizon_hours - lead_time
            deficits.append(deficit)
            n_hits += 1

        # 2. Miss (False Negative)
        elif true_idx is not None and pred_idx is None:
            deficits.append(missed_penalty)
            n_misses += 1

        # 3. False Alarm (False Positive)
        elif true_idx is None and pred_idx is not None:
            deficits.append(false_alarm_penalty)
            n_false_alarms += 1

        # 4. Quiet (True Negative)
        elif true_idx is None and pred_idx is None:
            # deficits.append(0.0)
            n_true_negatives += 1

    # DEBUG PRINT
    if len(y_true_batch) > 0:
        max_true_g = np.max(true_grads)
        max_pred_g = np.max(pred_grads)
        print(
            f"\n[DEBUG] Max True Grad: {max_true_g:.4f} | Max Pred Grad: {max_pred_g:.4f} | Thresh: {threshold}"
        )
        print(
            f"[DEBUG] Stats -> Hits: {n_hits}, Miss: {n_misses}, FA: {n_false_alarms}, TN: {n_true_negatives}"
        )

    return np.mean(deficits) if deficits else 0.0

def calculate_log_horizon_deficit(
    y_true_batch, y_pred_batch, threshold, sust_time,
    horizon_hours=12.0, eps=1e-3, event_only=True
):
    """
    Log-scaled distance from the ideal H-hour warning.
    Goal: MINIMIZE this (0 is perfect; larger = worse).

    For sequences with an event:
        penalty = -log(lead_time / H), with lead_time in (0, H]
    For misses / late detections: lead_time ~ 0 -> huge penalty.
    For false alarms: use a large fixed penalty.

    If event_only=True, we average only over sequences where an event actually occurs.
    """
    penalties = []
    seq_len = y_true_batch.shape[1]
    hours_per_step = horizon_hours / seq_len

    # How bad is "worst possible"?
    # Treat L ≈ eps as "virtually no lead":
    max_penalty = -np.log(eps / horizon_hours)

    for i in range(len(y_true_batch)):
        true_idx = get_first_emergence_index(y_true_batch[i], threshold, sust_time)
        pred_idx = get_first_emergence_index(y_pred_batch[i], threshold, sust_time)

        # No event in the true signal
        if true_idx is None:
            if event_only:
                # Skip TN/FA in this metric when event_only=True
                if pred_idx is not None:
                    # You can optionally track FA penalties separately
                    penalties.append(max_penalty)  # or skip & log FAs separately
                continue
            else:
                # Non-event sequences also count:
                if pred_idx is None:
                    penalties.append(0.0)  # quiet when quiet
                else:
                    penalties.append(max_penalty)  # false alarm
            continue

        # There *is* an event:
        true_time = true_idx * hours_per_step

        if pred_idx is None:
            # Missed: no lead time
            lead_time = 0.0
        else:
            pred_time = pred_idx * hours_per_step
            # Lead time cannot be negative; if late, treat as 0
            lead_time = max(true_time - pred_time, 0.0)

        # Make sure 0 < lead_time <= H for the log
        lead_time = np.clip(lead_time, eps, horizon_hours)

        # Log-scaled penalty
        penalty = -np.log(lead_time / horizon_hours)
        penalties.append(penalty)

    if len(penalties) == 0:
        return 0.0  # or np.nan, depending on how you want to handle "no events"

    return np.mean(penalties)


def calculate_exponential_deficit(y_true_batch, y_pred_batch, threshold, sust_time, horizon_hours=12.0):
    """
    Calculates deficit using an EXPONENTIAL penalty.
    Small deviations are okay, but missing the 12h mark gets expensive fast.
    """
    deficits = []
    
    # 1. Calculate Gradients
    true_grads = np.gradient(y_true_batch, axis=1)
    pred_grads = np.gradient(y_pred_batch, axis=1)
    
    seq_len = true_grads.shape[1]
    hours_per_step = horizon_hours / seq_len
    
    # --- TUNING THE EXPONENTIAL CURVE ---
    # A scale of 0.5 makes the curve steep but prevents math overflow.
    # Formula: Cost = exp(Deficit * scale) - 1
    scale = 0.5 
    
    # Calculate the max possible cost (Deficit = 12 hours -> Lead Time 0)
    # exp(12 * 0.5) - 1 = exp(6) - 1 ≈ 402.4
    max_horizon_cost = np.exp(horizon_hours * scale) - 1

    # --- PENALTY LOGIC ---
    # CRITICAL: Miss Penalty must be HIGHER than the cost of a late prediction
    # to force the model to attempt predictions.
    missed_penalty = max_horizon_cost * 1.5  # ~600.0 cost
    
    # False Alarm should be cheaper than a Miss to encourage sensitivity
    false_alarm_penalty = max_horizon_cost * 0.5 # ~200.0 cost

    n_hits = 0
    n_misses = 0
    n_false_alarms = 0
    n_true_negatives = 0

    for i in range(len(true_grads)):
        true_idx = get_first_emergence_index(true_grads[i], threshold, sust_time)
        pred_idx = get_first_emergence_index(pred_grads[i], threshold, sust_time)
        
        # 1. Hit
        if true_idx is not None and pred_idx is not None:
            true_time = true_idx * hours_per_step
            pred_time = pred_idx * hours_per_step
            
            lead_time = true_time - pred_time
            
            # Linear Deficit: 12 - Lead Time
            # If Lead is 12, Deficit is 0.
            # If Lead is 0, Deficit is 12.
            # If Lead is -2 (Late), Deficit is 14.
            lin_deficit = horizon_hours - lead_time
            
            # Exponential Cost Calculation
            # We use absolute deficit to ensure punishment goes both ways, 
            # but we technically only care about being late or too early.
            # Given your prompt: "farther from 12h = more exponential"
            
            # If lin_deficit is negative (e.g. predicted 14 hours before), treat as positive error
            abs_deficit = abs(lin_deficit)
            
            exp_cost = np.exp(abs_deficit * scale) - 1
            deficits.append(exp_cost)
            n_hits += 1

        # 2. Miss (False Negative)
        elif true_idx is not None and pred_idx is None:
            deficits.append(missed_penalty)
            n_misses += 1

        # 3. False Alarm (False Positive)
        elif true_idx is None and pred_idx is not None:
            deficits.append(false_alarm_penalty)
            n_false_alarms += 1

        # 4. Quiet (True Negative)
        elif true_idx is None and pred_idx is None:
            deficits.append(0.0) 
            n_true_negatives += 1

    # DEBUG PRINT
    if len(y_true_batch) > 0:
        max_true_g = np.max(true_grads)
        max_pred_g = np.max(pred_grads)
        print(f"\n[DEBUG] Max True Grad: {max_true_g:.4f} | Max Pred Grad: {max_pred_g:.4f} | Thresh: {threshold}")
        print(f"[DEBUG] Stats -> Hits: {n_hits}, Miss: {n_misses}, FA: {n_false_alarms}, TN: {n_true_negatives}")
        print(f"[DEBUG] Avg Exp Cost: {np.mean(deficits) if deficits else 0:.2f}")

    return np.mean(deficits) if deficits else 0.0

def validate_modelv2(
    model, dataloader, device, threshold=THRESHOLD, sust_time=SUST_TIME
):
    model.eval()
    all_preds = []
    all_y = []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            all_preds.append(preds.cpu().numpy())
            all_y.append(y.cpu().numpy())

    np_preds = np.concatenate(all_preds, axis=0)
    np_y = np.concatenate(all_y, axis=0)

    # 1. Standard RMSE
    pred_derivatives = np.diff(np_preds, axis=1)
    true_derivatives = np.diff(np_y, axis=1)
    deriv_rmse = np.sqrt(np.mean((pred_derivatives - true_derivatives) ** 2))

    # 2. Horizon Deficit (LOWER IS BETTER)
    log_deficit = calculate_exponential_deficit(
        np_y, np_preds, threshold, sust_time,
        horizon_hours=12.0
    )

    return deriv_rmse, log_deficit


def validate_modelv1(
    model, dataloader, device, threshold=THRESHOLD, sust_time=SUST_TIME
):
    model.eval()
    all_preds = []
    all_y = []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            all_preds.append(preds.cpu().numpy())
            all_y.append(y.cpu().numpy())

    np_preds = np.concatenate(all_preds, axis=0)
    np_y = np.concatenate(all_y, axis=0)

    # 1. Derivative RMSE (Keep this to ensure signal shape is generally correct)
    pred_derivatives = np.diff(np_preds, axis=1)
    true_derivatives = np.diff(np_y, axis=1)
    deriv_rmse = np.sqrt(np.mean((pred_derivatives - true_derivatives) ** 2))

    # 2. Mean Lead Time Score (The metric you actually care about)
    # returns a value in HOURS. (e.g., 4.5 means on average 4.5 hours early)
    avg_lead_time = calculate_lead_time_score(
        np_y, np_preds, threshold, sust_time, horizon_hours=12.0
    )

    print(f"Validation Results:")
    print(f"  Signal Shape Error (RMSE): {deriv_rmse:.4f}")
    print(f"  Avg Lead Time (Hours):     {avg_lead_time:.4f} (Higher is better)")

    return deriv_rmse, avg_lead_time


def validate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_y = []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            preds = model(x)

            all_preds.append(preds.cpu().numpy())
            all_y.append(y.cpu().numpy())

    # (The rest of the derivative RMSE calculation is the same)
    # ...
    pred_derivatives = np.diff(np.concatenate(all_preds, axis=0), axis=1)
    true_derivatives = np.diff(np.concatenate(all_y, axis=0), axis=1)
    derivative_rmse = np.sqrt(np.mean((pred_derivatives - true_derivatives) ** 2))

    return derivative_rmse


class PlateauStopper(tune.stopper.Stopper):
    """Stops trials when the metric has plateaued."""

    def __init__(
        self,
        metric: str,
        min_epochs: int = 20,
        patience: int = 10,
        min_improvement_percent: float = 1e-5,
    ):
        """
        Args:
            metric: The metric to monitor.
            min_epochs: Minimum number of epochs to run before stopping is considered.
            patience: Number of recent epochs to check for improvement.
            min_improvement: The minimum improvement required to not stop the trial.
        """
        self._metrics = metric
        self._min_epochs = min_epochs
        self._patience = patience
        self._min_improvement = min_improvement_percent
        self._trial_history = {}  # To store the history of each trial

    def __call__(self, trial_id: str, result: dict) -> bool:
        """This is called after each tune.report() call."""
        # Initialize history for a new trial
        if trial_id not in self._trial_history:
            self._trial_history[trial_id] = []

        history = self._trial_history[trial_id]
        history.append(result[self._metrics])

        # Don't stop if we haven't reached the minimum number of epochs
        if len(history) <= self._min_epochs:
            return False

        # Check for improvement over the patience window
        # We look at the best value in the last `patience` epochs
        # and compare it to the best value before that window.
        window = history[-self._patience :]
        previous_best = min(history[: -self._patience])
        current_best = min(window)

        # If there's no meaningful improvement, stop the trial
        improvement_needed = previous_best * self._min_improvement / 100
        if previous_best - current_best < improvement_needed:
            print(
                f"Stopping trial {trial_id}: "
                f"No improvement of {improvement_needed} in the last {self._patience} epochs."
            )
            return True

        return False

    def stop_all(self) -> bool:
        """This function is used to stop all trials at once. We don't need it here."""
        return False

class WeightedMSELoss(nn.Module):
    def __init__(self, high_val_weight=50.0, threshold=0.01):
        super(WeightedMSELoss, self).__init__()
        self.high_val_weight = high_val_weight
        self.threshold = threshold

    def forward(self, input, target):
        # Calculate standard squared error
        loss = (input - target) ** 2
        
        # Create a weight map
        # If the TARGET has a high gradient/value, multiply the loss by 50
        # This forces the model to pay attention to the events
        
        # Option A: Weight by Value Magnitude
        weights = torch.ones_like(loss)
        weights[target > self.threshold] = self.high_val_weight
        
        # Option B (Better for you): Weight by Gradient Magnitude
        # (Requires computing gradient inside the loss, can be slow, 
        #  so Option A is usually a good proxy if peaks are high value)
        
        return torch.mean(loss * weights)
    

def add_grid_lines(ax, divisions=9, color="w", linewidth=1):
    """
    Adds grid lines to an image plot to visually divide it into a matrix.

    Parameters:
    - ax: The axes object to add grid lines to.
    - divisions: Number of divisions along each axis (default is 9 for a 9x9 grid).
    - color: Color of the grid lines.
    - linewidth: Width of the grid lines.
    """
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    x_spacing = np.linspace(xlim[0], xlim[1], divisions + 1)
    y_spacing = np.linspace(ylim[0], ylim[1], divisions + 1)

    for x in x_spacing[1:-1]:
        ax.axvline(x=x, color=color, linewidth=linewidth)
    for y in y_spacing[1:-1]:
        ax.axhline(y=y, color=color, linewidth=linewidth)
