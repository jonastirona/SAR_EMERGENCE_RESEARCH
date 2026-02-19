# functions to be used in pipeline
from datetime import datetime
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
# Setup project paths
current_dir = os.path.dirname(os.path.abspath(__file__))
BASE_PROJECT_DIR = os.path.normpath(os.path.join(current_dir, ".."))

DATA_PATH = os.path.join(BASE_PROJECT_DIR, "data")
RESULTS_PATH = os.path.join(current_dir, "results") + "/"
MODELS_PATH = os.path.join(current_dir, "models")


##### Sept 18th and later
def min_max_scaling(arr, min_val, max_val):
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
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
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


def split_sequences(input_sequences, output_sequences, n_steps_in, n_steps_out):
    X, y = list(), list()
    last_vals = list()
    for i in range(len(input_sequences)):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out - 1
        if out_end_ix > len(input_sequences):
            break
        seq_x, seq_y = (
            input_sequences[i:end_ix],
            output_sequences[end_ix - 1 : out_end_ix],
        )
        # Store last value for RMSE calibration on validation
        last_val = output_sequences[end_ix - 2]
        X.append(seq_x), y.append(seq_y), last_vals.append(last_val)
    return np.array(X), np.array(y), np.array(last_vals)


def calculate_metrics(timeline_true, timeline_predicted):
    timeline_true = np.array(timeline_true)
    timeline_predicted = np.array(timeline_predicted)
    MAE = np.mean(np.abs(timeline_predicted - timeline_true))
    MSE = np.mean(np.square(timeline_predicted - timeline_true))
    RMSE = np.sqrt(MSE)
    RMSLE = np.sqrt(
        np.mean(np.square(np.log1p(timeline_predicted) - np.log1p(timeline_true)))
    )
    SS_res = np.sum(np.square(timeline_true - timeline_predicted))
    SS_tot = np.sum(np.square(timeline_true - np.mean(timeline_true)))
    R_squared = 1 - (SS_res / SS_tot)
    return MAE, MSE, RMSE, RMSLE, R_squared


def emergence_indication(d_true, threshold, sust_time):
    d_true = smooth_with_numpy(d_true)
    indicator = np.zeros(d_true.shape)
    for j in range(len(d_true)):
        if d_true[j] >= threshold:
            indicator[j] = 1

    # Enforce sustained condition
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


def smooth_with_numpy(d_true, window_size=5):
    if window_size <= 1:
        return d_true
    pad_width = window_size // 2
    padded_d_true = np.pad(d_true, pad_width, mode="edge")
    window = np.ones(window_size) / window_size
    smoothed_d_true = np.convolve(padded_d_true, window, mode="same")
    return smoothed_d_true[pad_width:-pad_width] if pad_width else smoothed_d_true


def recalibrate(pred, previous_value):
    trend = pred - pred[0]
    new_pred = trend + previous_value
    return new_pred


def highlight_tile(ax, tile_number, divisions=9, color="r", linewidth=1):
    """Highlights a specific tile in the grid."""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    tile_width = (xlim[1] - xlim[0]) / divisions
    tile_height = (ylim[1] - ylim[0]) / divisions

    row_idx = (tile_number - 1) // divisions
    col_idx = (tile_number - 1) % divisions

    x = xlim[0] + col_idx * tile_width
    y = ylim[1] - (row_idx + 1) * tile_height

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
        pm_path = os.path.join(DATA_PATH, f"AR{ar_num}", f"mean_pmdop{ar_num}_flat.npz")
        mag_path = os.path.join(DATA_PATH, f"AR{ar_num}", f"mean_mag{ar_num}_flat.npz")
        int_path = os.path.join(DATA_PATH, f"AR{ar_num}", f"mean_int{ar_num}_flat.npz")

        power_maps = np.load(pm_path, allow_pickle=True)
        mag_flux_data = np.load(mag_path, allow_pickle=True)
        intensities_data = np.load(int_path, allow_pickle=True)
        time = power_maps["arr_4"]

        power_maps = [power_maps[f"arr_{i}"] for i in range(4)]
        mag_flux = mag_flux_data["arr_0"]
        intensities = intensities_data["arr_0"]

        # Trim top/bottom rows and handle NaNs
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
    )
    if not matches:
        raise Exception("Could not parse parameters from filename", filename)

    # Parse num_layers, hidden_size, learning_rate, dropout
    num_layers, hidden_size, learning_rate, dropout = [
        float(val) if i >= 2 else int(val) for i, val in enumerate(matches[0])
    ]
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

                # Assign weights based on labeled tiles from scales.json
                real_tile_idx = tile + rid_of_top * size
                if ar_list is not None and isinstance(tile_weights, dict):
                    current_ar = str(ar_list[i])
                    w = 0.05  # Default background weight
                    if current_ar in tile_weights:
                        # tiles in json are 1-based indices
                        if (real_tile_idx + 1) in tile_weights[current_ar]:
                            w = 1.0
                    else:
                        w = 0.05
                else:
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


# --- Model Training & Evaluation ---
def train_epochHybridLSTM(
    model, dataloader, loss_fn, optimizer, device, teacher_ratio, alpha
):
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

        outputs = model(x, y, teacher_forcing_ratio=teacher_ratio)

        # Loss on values
        if weights is not None:
            value_loss = (torch.mean((outputs - y) ** 2, dim=1) * weights).mean()
        else:
            value_loss = loss_fn(outputs, y)

        # Loss on gradients (matching np.gradient in test)
        outputs_grad = torch.gradient(outputs, dim=1)[0]
        y_grad = torch.gradient(y, dim=1)[0]

        if weights is not None:
            gradient_loss = (
                torch.mean((outputs_grad - y_grad) ** 2, dim=1) * weights
            ).mean()
        else:
            gradient_loss = loss_fn(outputs_grad, y_grad)

        # Combine weighted losses
        loss = alpha * value_loss + (1 - alpha) * gradient_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def train_epochHybridVanillaLSTM(model, dataloader, loss_fn, optimizer, device, alpha):
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

        # 1. Calculate loss on the actual values
        if weights is not None:
            value_loss = (torch.mean((outputs - y) ** 2, dim=1) * weights).mean()
        else:
            value_loss = loss_fn(outputs, y)

        # 2. Calculate loss on the gradients (central differences, matching np.gradient in test)
        outputs_grad = torch.gradient(outputs, dim=1)[0]
        y_grad = torch.gradient(y, dim=1)[0]

        if weights is not None:
            gradient_loss = (
                torch.mean((outputs_grad - y_grad) ** 2, dim=1) * weights
            ).mean()
        else:
            gradient_loss = loss_fn(outputs_grad, y_grad)

        # 3. Combine: alpha=1.0 -> value only, alpha=0.0 -> gradient only
        loss = alpha * value_loss + (1 - alpha) * gradient_loss

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

    # 2. Gradient RMSE (central differences, matching np.gradient in test)
    pred_gradients = np.gradient(all_preds_np, axis=1)
    true_gradients = np.gradient(all_y_np, axis=1)
    gradient_rmse = np.sqrt(np.mean((pred_gradients - true_gradients) ** 2))

    # 3. Weighted RMSE & Weighted Gradient RMSE
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

        # Weighted Gradient RMSE
        w_grad_mse = np.average(
            (pred_gradients - true_gradients) ** 2,
            weights=np.broadcast_to(weights_expanded, pred_gradients.shape),
        )
        w_grad_rmse = np.sqrt(w_grad_mse)

        # 4. Active vs Background RMSE (split by weight)
        active_mask = all_weights_np == 1.0
        bg_mask = all_weights_np < 1.0

        if np.any(active_mask):
            active_rmse = np.sqrt(
                np.mean((all_preds_np[active_mask] - all_y_np[active_mask]) ** 2)
            )
            active_grad_rmse = np.sqrt(
                np.mean(
                    (pred_gradients[active_mask] - true_gradients[active_mask]) ** 2
                )
            )
        else:
            active_rmse = rmse
            active_grad_rmse = gradient_rmse

        if np.any(bg_mask):
            bg_rmse = np.sqrt(np.mean((all_preds_np[bg_mask] - all_y_np[bg_mask]) ** 2))
        else:
            bg_rmse = rmse
    else:
        w_rmse = rmse
        w_grad_rmse = gradient_rmse
        active_rmse = rmse
        active_grad_rmse = gradient_rmse
        bg_rmse = rmse

    return {
        "RMSE": rmse,
        "Grad_RMSE": gradient_rmse,
        "Weighted_RMSE": w_rmse,
        "Weighted_Grad_RMSE": w_grad_rmse,
        "Active_RMSE": active_rmse,
        "Active_Grad_RMSE": active_grad_rmse,
        "Background_RMSE": bg_rmse,
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
        if trial_id not in self._trial_history:
            self._trial_history[trial_id] = []

        history = self._trial_history[trial_id]
        history.append(result[self._metrics])

        if len(history) <= self._min_epochs:
            return False

        # Check for improvement over the patience window
        window = history[-self._patience :]
        previous_best = min(history[: -self._patience])
        current_best = min(window)

        improvement_needed = previous_best * self._min_improvement / 100
        if previous_best - current_best < improvement_needed:
            print(f"Stopping trial {trial_id}: plateau reached.")
            return True

        return False

    def stop_all(self) -> bool:
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
