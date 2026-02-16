# functions to be used in pipeline

from datetime import datetime, timedelta
import numpy as np
import os
import torch
import torch.nn as nn
import warnings
import random
from ray import tune
import re

isVanillaLSTM = True
if isVanillaLSTM:
    model_type = "VanillaLSTM"
else:
    model_type = "LSTM"
warnings.filterwarnings("ignore")
# Determine paths relative to this file
current_dir = os.path.dirname(os.path.abspath(__file__))
# Assumes structure: project/lstm/functions.py
#                  project/data/
#                  project/models/
#                  project/lstm/results/
BASE_PROJECT_DIR = os.path.normpath(os.path.join(current_dir, ".."))

DATA_PATH = os.path.join(BASE_PROJECT_DIR, "data")
RESULTS_PATH = os.path.join(current_dir, "results") + "/"
MODELS_PATH = os.path.join(current_dir, "models")


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

    # Add tile number alternating between top and bottom
    text_y = y + 1.2 if (x + y) % 2 == 0 else y + -0.35
    ax.text(
        x + 0.5,
        text_y,
        str(tile_number),
        ha="center",
        va="center",
        fontsize=11,
        color="black",
        weight="bold",
        zorder=2,
    )


def load_ar_data(ar_num, size, rid_of_top):
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
        trim_slice = slice(rid_of_top * size, size**2 - rid_of_top * size)
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


def load_all_ar_data(ar_list, size, rid_of_top):
    all_power_maps = []
    all_flux = []
    all_cont_int = []
    for ar in ar_list:
        power_maps, flux, cont_int, time = load_ar_data(ar, size, rid_of_top)
        if power_maps is not None:
            all_power_maps.append(power_maps)
            all_flux.append(flux)
            all_cont_int.append(cont_int)
    return all_power_maps, all_flux, all_cont_int


def scale_and_combine_data(maps, flux, cont_int, m_scale, f_scale, cont_int_scale):
    stacked_maps = min_max_scaling(maps, *m_scale)
    mag_flux = min_max_scaling(flux, *f_scale)
    intensities = min_max_scaling(cont_int, *cont_int_scale)

    # Reshape int to have an extra dimension and then put it with pmaps
    int_reshaped = np.expand_dims(intensities, axis=1)

    inputs = np.concatenate([stacked_maps, int_reshaped], axis=1)

    return inputs, mag_flux


def get_params(filename):
    model_type = "LSTM"
    if "VanillaLSTM" in filename:
        model_type = "VanillaLSTM"
    elif "LSTM" in filename:
        pass
    else:
        raise Exception("UNKNOWN NAME", filename)

    matches = re.findall(
        r"_n(\d+)_h(\d+)_lr([0-9.]+)_d([0-9.]+)",
        filename,
    )  # Extract numbers from the filename
    if not matches:
        raise Exception("Could not parse parameters from filename", filename)
    (
        num_layers,
        hidden_size,
        learning_rate,
        dropout,
    ) = [
        float(val) if i >= 2 else int(val) for i, val in enumerate(matches[0])
    ]  # Unpack the matched values into variables
    return (
        model_type,
        num_layers,
        hidden_size,
        learning_rate,
        dropout,
    )


def AR_defs(val_AR):
    before_plot, num_in, NOAA_first, NOAA_second = None, None, None, None
    end = 0
    start = 0
    if val_AR == 11698:
        window_s = 74
        starting_tile = 45
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
    power_maps_scale=None,
    flux_scale=None,
    cont_int_scale=None,
    tile_weights=None,  # Added tile_weights parameter
    pre_loaded_data=None,  # Added pre_loaded_data parameter
):
    """Builds a complete dataset (X, y) for a list of ARs."""
    processed_inputs, processed_flux = [], []
    all_power_maps = []
    all_flux = []
    all_cont_int = []

    # Load data for all ARs
    if pre_loaded_data is not None:
        all_power_maps, all_flux, all_cont_int = pre_loaded_data
    elif ar_list is not None:
        all_power_maps, all_flux, all_cont_int = load_all_ar_data(
            ar_list, size, rid_of_top
        )

    if tile_weights is not None:
        print(f"Applying AR-specific tile weights. Labeled: 1.0, Unlabeled: 0.3")
    else:
        print("No tile weights provided. Using default weight 1.0.")

    if power_maps_scale is None:
        power_maps_scale = (np.min(all_power_maps), np.max(all_power_maps))
        flux_scale = (np.min(all_flux), np.max(all_flux))
        cont_int_scale = (np.min(all_cont_int), np.max(all_cont_int))

    for i in range(len(all_power_maps)):
        combined_inputs, flux = scale_and_combine_data(
            all_power_maps[i],
            all_flux[i],
            all_cont_int[i],
            power_maps_scale,
            flux_scale,
            cont_int_scale,
        )
        processed_inputs.append(combined_inputs)
        processed_flux.append(flux)

    if not processed_inputs:
        print("all_inputs_list does not exist")
        return None, None, 0

    # Create sequences for the LSTM
    x_list, y_list, last_list, weight_list, tile_idx_list = [], [], [], [], []
    tiles = size**2 - 2 * size * rid_of_top

    for i, (inputs, flux) in enumerate(zip(processed_inputs, processed_flux)):
        for tile in range(tiles):
            x_seq, y_seq, last_seq = lstm_ready(
                tile, size, inputs, flux, num_in, num_pred
            )
            if x_seq.shape[0] > 0:
                x_seq = torch.reshape(x_seq, (x_seq.shape[0], num_in, x_seq.shape[2]))
                x_list.append(x_seq)
                y_list.append(y_seq)
                last_list.append(last_seq)

                # Assign weights to each sample in the sequence based on the tile
                real_tile_idx = tile + rid_of_top * size

                # Check if we have an AR list and a weight dict
                if ar_list is not None and isinstance(tile_weights, dict):
                    # Get the current AR number
                    current_ar = str(
                        ar_list[i]
                    )  # ar_list corresponds to processed_inputs

                    # Default weight for non-labeled tiles
                    w = 0.05

                    # Check if AR is in our labeled regions
                    if current_ar in tile_weights:
                        # tile_weights[current_ar] is a list of ACTIVE tiles (1-based index usually, let's verify)
                        # The json had indices like 39, 40 etc. Let's assume these are 1-based as per user request in other contexts?
                        # "labeled regions in region: [tile_num (index 1)] json format" -> YES, 1-based.
                        # real_tile_idx is 0-based. So we compare real_tile_idx + 1 with the list.
                        if (real_tile_idx + 1) in tile_weights[current_ar]:
                            w = 1.0
                    else:
                        # If AR not in dict, maybe default to 0.3 or 1.0?
                        # User said: "calculate loss with proportions 0.3 for non labeled tiles and 1 for labeled tiles."
                        # Implies if not labeled, it's non-labeled -> 0.3
                        w = 0.05
                else:
                    # Fallback or Validation case (if tile_weights is None)
                    w = 1.0

                weight_list.append(torch.full((x_seq.shape[0],), float(w)))
                tile_idx_list.append(
                    torch.full((x_seq.shape[0],), float(real_tile_idx))
                )

    if not x_list:
        print("X_list does not exist")
        return None, None, 0

    x_all = torch.cat(x_list, dim=0)
    y_all = torch.cat(y_list, dim=0)
    last_all = torch.cat(last_list, dim=0)
    weights_all = torch.cat(weight_list, dim=0)
    tile_indices_all = torch.cat(tile_idx_list, dim=0)
    input_feature_size = x_all.shape[2]

    return (
        x_all,
        y_all,
        last_all,
        weights_all,
        tile_indices_all,
        input_feature_size,
        power_maps_scale,
        flux_scale,
        cont_int_scale,
    )


# --- Emergence-Aware Loss Components ---
def differentiable_smooth(x, window_size=5):
    """Differentiable 1D smoothing using convolution.
    Args:
        x: tensor of shape (batch, timesteps)
        window_size: smoothing window (must be odd)
    Returns:
        smoothed tensor of same shape
    """
    if window_size <= 1:
        return x
    kernel = torch.ones(1, 1, window_size, device=x.device, dtype=x.dtype) / window_size
    # Pad to maintain length
    pad = window_size // 2
    # x shape: (batch, timesteps) -> (batch, 1, timesteps) for conv1d
    x_padded = torch.nn.functional.pad(x.unsqueeze(1), (pad, pad), mode="replicate")
    smoothed = torch.nn.functional.conv1d(x_padded, kernel).squeeze(1)
    return smoothed


def soft_emergence_time(derivatives, threshold=0.01, k=20.0, window_size=5):
    """Compute differentiable emergence onset time using rising-edge detection.

    Instead of finding the centroid of the sustained indicator (which would
    misrepresent onset for long emergences), this isolates the rising edge
    of the soft threshold crossing.

    Args:
        derivatives: tensor (batch, timesteps) — discrete derivatives of the signal
        threshold: emergence threshold
        k: sigmoid steepness (annealed during training)
        window_size: smoothing window size
    Returns:
        soft_times: tensor (batch,) — differentiable onset time per sample
        has_emergence: tensor (batch,) — soft indicator of whether emergence exists
    """
    # 1. Smooth the derivatives
    smoothed = differentiable_smooth(derivatives, window_size)

    # 2. Soft threshold: sigmoid approximation of (smoothed >= threshold)
    soft_ind = torch.sigmoid(k * (smoothed - threshold))

    # 3. Rising edge detection: diff of soft indicator, keep only positive (onsets)
    soft_edge = soft_ind[:, 1:] - soft_ind[:, :-1]
    rising = torch.relu(soft_edge)  # Only keep positive transitions (onsets)

    # 4. Weighted average of rising edge to find onset time
    timesteps = torch.arange(rising.shape[1], device=rising.device, dtype=rising.dtype)
    timesteps = timesteps.unsqueeze(0).expand_as(rising)  # (batch, T-1)

    # Sum of rising edge weights (how much emergence is detected)
    rising_sum = rising.sum(dim=1) + 1e-8  # avoid division by zero

    # Weighted onset time
    soft_times = (timesteps * rising).sum(dim=1) / rising_sum

    # "has emergence" indicator: is the total rising signal significant?
    has_emergence = torch.sigmoid(10.0 * (rising.sum(dim=1) - 0.1))

    return soft_times, has_emergence


def emergence_timing_loss(pred, y, threshold=0.01, k=20.0):
    """Compute emergence timing loss between predicted and true sequences.

    Uses rising-edge detection to isolate onset times, then penalizes
    squared difference. Quiet samples (no emergence in either) contribute 0.

    Args:
        pred: tensor (batch, num_pred) — predicted values
        y: tensor (batch, num_pred) — ground truth values
        threshold: emergence threshold for derivative
        k: sigmoid steepness (annealed via curriculum)
    Returns:
        scalar loss
    """
    # Compute derivatives (discrete differences)
    d_pred = pred[:, 1:] - pred[:, :-1]
    d_true = y[:, 1:] - y[:, :-1]

    # Get soft emergence onset times
    t_pred, has_pred = soft_emergence_time(d_pred, threshold, k)
    t_true, has_true = soft_emergence_time(d_true, threshold, k)

    # Only penalize when both have emergence (avoid penalizing quiet samples)
    # Soft AND: multiply the two indicators
    # Detach mask so the model can't learn to "hide" emergence to avoid penalty
    both_have = (has_pred * has_true).detach()

    # Squared timing error, weighted by whether both have emergence
    timing_error = (t_pred - t_true) ** 2
    loss = (timing_error * both_have).sum() / (both_have.sum() + 1e-8)

    return loss


# --- Model Training & Evaluation ---
def train_epochHybridLSTM(
    model, dataloader, loss_fn, optimizer, device, teacher_ratio, alpha
):
    model.train()
    total_loss = 0
    loss_scaler = 100.0

    for batch in dataloader:
        if len(batch) == 3:
            x, y, weights = batch
            weights = weights.to(device)
        else:
            x, y = batch
            weights = None

        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        outputs = model(x, y, teacher_forcing_ratio=teacher_ratio)

        # 1. Calculate loss on the actual values
        if weights is not None:
            # Manually calculate weighted MSE
            value_loss = (torch.mean((outputs - y) ** 2, dim=1) * weights).mean()
        else:
            value_loss = loss_fn(outputs, y)

        # 2. Calculate loss on the derivatives
        outputs_deriv = outputs[:, 1:] - outputs[:, :-1]
        y_deriv = y[:, 1:] - y[:, :-1]

        if weights is not None:
            derivative_loss = (
                torch.mean((outputs_deriv - y_deriv) ** 2, dim=1) * weights
            ).mean()
        else:
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

    for batch in dataloader:
        if len(batch) == 3:
            x, y, weights = batch
            weights = weights.to(device)
        else:
            x, y = batch
            weights = None

        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        outputs = model(x, y, teacher_forcing_ratio=teacher_forcing)

        if weights is not None:
            # Weighted MSE loss: (pred - true)^2 * weights
            loss = (torch.mean((outputs - y) ** 2, dim=1) * weights).mean()
        else:
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

    for batch in dataloader:
        if len(batch) == 3:
            x, y, weights = batch
            weights = weights.to(device)
        else:
            x, y = batch
            weights = None

        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        outputs = model(x)

        # 1. Calculate loss on the actual values
        if weights is not None:
            value_loss = (torch.mean((outputs - y) ** 2, dim=1) * weights).mean()
        else:
            value_loss = loss_fn(outputs, y)

        # 2. Calculate loss on the derivatives
        outputs_deriv = outputs[:, 1:] - outputs[:, :-1]
        y_deriv = y[:, 1:] - y[:, :-1]

        if weights is not None:
            derivative_loss = (
                torch.mean((outputs_deriv - y_deriv) ** 2, dim=1) * weights
            ).mean()
        else:
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

    for batch in dataloader:
        if len(batch) == 3:
            x, y, weights = batch
            weights = weights.to(device)
        else:
            x, y = batch
            weights = None

        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        outputs = model(x)

        if weights is not None:
            loss = (torch.mean((outputs - y) ** 2, dim=1) * weights).mean()
        else:
            loss = loss_fn(outputs, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def train_epoch_emergence_aware(
    model, dataloader, loss_fn, optimizer, device, alpha, gamma, k, teacher_forcing=None
):
    """Training epoch with emergence-aware loss.

    Combines:
      - Value MSE (weight alpha)
      - Derivative MSE (weight beta = (1-alpha)*(1-gamma))
      - Emergence timing loss (weight gamma)

    Works for both LSTM (with teacher_forcing) and VanillaLSTM (teacher_forcing=None).
    """
    model.train()
    total_loss = 0
    beta = (1.0 - alpha) * (1.0 - gamma)

    for batch in dataloader:
        if len(batch) == 3:
            x, y, weights = batch
            weights = weights.to(device)
        else:
            x, y = batch
            weights = None

        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        # Forward pass (teacher forcing for encoder-decoder LSTM)
        if teacher_forcing is not None:
            outputs = model(x, y, teacher_forcing_ratio=teacher_forcing)
        else:
            outputs = model(x)

        # 1. Value MSE
        if weights is not None:
            value_loss = (torch.mean((outputs - y) ** 2, dim=1) * weights).mean()
        else:
            value_loss = loss_fn(outputs, y)

        # 2. Derivative MSE
        d_out = outputs[:, 1:] - outputs[:, :-1]
        d_y = y[:, 1:] - y[:, :-1]
        if weights is not None:
            derivative_loss = (torch.mean((d_out - d_y) ** 2, dim=1) * weights).mean()
        else:
            derivative_loss = loss_fn(d_out, d_y)

        # 3. Emergence timing loss
        timing_loss = emergence_timing_loss(outputs, y, threshold=0.01, k=k)

        # Combined loss
        loss = alpha * value_loss + beta * derivative_loss + gamma * timing_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_y = []
    all_weights = []

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                x, y, weights = batch
            else:
                x, y = batch
                weights = None

            x, y = x.to(device), y.to(device)

            preds = model(x)

            all_preds.append(preds.cpu().numpy())
            all_y.append(y.cpu().numpy())
            if weights is not None:
                all_weights.append(weights.cpu().numpy())

    all_preds_np = np.concatenate(all_preds, axis=0)
    all_y_np = np.concatenate(all_y, axis=0)

    # 1. Standard RMSE (Value)
    mse = np.mean((all_preds_np - all_y_np) ** 2)
    rmse = np.sqrt(mse)

    # 2. Derivative RMSE
    pred_derivatives = np.diff(all_preds_np, axis=1)
    true_derivatives = np.diff(all_y_np, axis=1)
    derivative_rmse = np.sqrt(np.mean((pred_derivatives - true_derivatives) ** 2))

    # 3. Weighted RMSE & Weighted Derivative RMSE
    if all_weights:
        all_weights_np = np.concatenate(all_weights, axis=0)
        # Weights are (batch,). Need to broadcast to (batch, num_pred) or just average over batch?
        # The loss function averages over batch AND time.
        # weights correspond to the sample (row). So we weight each row.
        # (batch, num_pred) -> average over axes.
        weights_expanded = all_weights_np[:, np.newaxis]
        # Weighted mean of squared errors
        w_mse = np.average(
            (all_preds_np - all_y_np) ** 2,
            weights=np.broadcast_to(weights_expanded, all_preds_np.shape),
        )
        w_rmse = np.sqrt(w_mse)

        # Weighted Derivative RMSE
        w_deriv_mse = np.average(
            (pred_derivatives - true_derivatives) ** 2,
            weights=np.broadcast_to(weights_expanded, pred_derivatives.shape),
        )
        w_deriv_rmse = np.sqrt(w_deriv_mse)

        # 4. Active vs Background RMSE (split by weight)
        active_mask = all_weights_np == 1.0
        bg_mask = all_weights_np < 1.0

        if np.any(active_mask):
            active_rmse = np.sqrt(
                np.mean((all_preds_np[active_mask] - all_y_np[active_mask]) ** 2)
            )
            active_deriv_rmse = np.sqrt(
                np.mean(
                    (pred_derivatives[active_mask] - true_derivatives[active_mask]) ** 2
                )
            )
        else:
            active_rmse = rmse
            active_deriv_rmse = derivative_rmse

        if np.any(bg_mask):
            bg_rmse = np.sqrt(np.mean((all_preds_np[bg_mask] - all_y_np[bg_mask]) ** 2))
        else:
            bg_rmse = rmse
    else:
        w_rmse = rmse
        w_deriv_rmse = derivative_rmse
        active_rmse = rmse
        active_deriv_rmse = derivative_rmse
        bg_rmse = rmse

    # 5. Emergence Timing MAE (non-differentiable, using existing emergence_indication)
    threshold = 0.01
    sust_time = 4
    timing_errors = []
    for i in range(len(all_preds_np)):
        d_pred_i = np.gradient(all_preds_np[i])
        d_true_i = np.gradient(all_y_np[i])
        ind_pred = emergence_indication(d_pred_i, threshold, sust_time)
        ind_true = emergence_indication(d_true_i, threshold, sust_time)

        # Find first emergence timestep for each
        t_pred_i = None
        t_true_i = None
        for t_idx in range(len(ind_pred)):
            if ind_pred[t_idx] == 1 and t_pred_i is None:
                t_pred_i = t_idx
            if ind_true[t_idx] == 1 and t_true_i is None:
                t_true_i = t_idx
            if t_pred_i is not None and t_true_i is not None:
                break

        # Only count if both have emergence
        if t_pred_i is not None and t_true_i is not None:
            timing_errors.append(abs(t_pred_i - t_true_i))

    emergence_timing_mae = np.mean(timing_errors) if timing_errors else 0.0

    return {
        "RMSE": rmse,
        "Deriv_RMSE": derivative_rmse,
        "Weighted_RMSE": w_rmse,
        "Weighted_Deriv_RMSE": w_deriv_rmse,
        "Active_RMSE": active_rmse,
        "Active_Deriv_RMSE": active_deriv_rmse,
        "Background_RMSE": bg_rmse,
        "Emergence_Timing_MAE": emergence_timing_mae,
    }


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
