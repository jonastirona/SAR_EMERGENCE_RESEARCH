import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from collections import OrderedDict
from datetime import datetime

# --- IMPORTANT: Make sure these imports point to your functions file ---
from functions import (
    lstm_ready,
    process_data,
    get_params,
    AR_defs,
    load_ar_data,
    LSTM as LSTM,
)


def initialize_lstm(
    inputs, hidden_size, num_layers, num_pred, state_dict, filename, device
):
    """Initializes the LSTM model and loads the trained weights."""
    input_size = np.shape(inputs)[1]
    lstm = LSTM(input_size, hidden_size, num_layers, num_pred).to(device)
    saved_state_dict = state_dict or torch.load(filename, map_location=device)

    # Handle models saved with DataParallel
    new_state_dict = OrderedDict()
    for k, v in saved_state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v

    lstm.load_state_dict(new_state_dict)
    lstm.eval()
    return lstm


def eval_AR_emergence_with_plots(
    device,
    save_fig,
    path,
    model_filename="pred12_r4_i110_n1_h2_e15_lr0.00100000_d0.2.pth",
    state_dict=None,
):
    """
    Evaluates a trained LSTM model, plotting the DIRECT SCALED output.
    NO inverse scaling is performed.
    """
    (
        num_pred,
        rid_of_top,
        num_in,
        num_layers,
        hidden_size,
        _,
        _,
        _,
        filename,
    ) = get_params(path, model_filename)

    # These scales MUST match the scales used during training
    scales_file = np.load(
        os.path.join(path, "SAR_EMERGENCE_RESEARCH/lstm/results", "model_scales.npz")
    )
    m_scale = scales_file["m_scale"]
    flux_scale = scales_file["flux_scale"]
    cont_int_scale = scales_file["cont_int_scale"]
    print(scales_file)
    print("Loaded scaling parameters from file.")

    for test_AR in [11698, 11726, 13165, 13179, 13183]:
        before_plot, _, _, _, starting_tile, _ = AR_defs(test_AR)
        if not before_plot:
            continue

        size = 9

        # Load data - assumes load_ar_data calculates the derivative
        maps, flux_derivative, cont_int, time = load_ar_data(
            test_AR, size, rid_of_top, starting_tile
        )
        inputs, target_scaled = process_data(
            maps,
            flux_derivative,
            cont_int,
            m_scale,
            flux_scale,
            cont_int_scale,
        )

        lstm = initialize_lstm(
            inputs, hidden_size, num_layers, num_pred, state_dict, filename, device
        )

        fig, axes = plt.subplots(4, 2, figsize=(14, 16), constrained_layout=True)
        axes = axes.flatten()

        future = 11  # The 12th hour prediction

        for i in range(7):  # Loop through 7 tiles
            ax = axes[i]
            tile_num_actual = starting_tile + i + 1
            print(f"\nProcessing Tile {tile_num_actual}")

            X_test, y_test_scaled, _ = lstm_ready(
                1 + i, size, inputs, target_scaled, num_in, num_pred
            )
            X_test = X_test.to(device)

            if X_test.shape[0] == 0:
                ax.text(
                    0.5,
                    0.5,
                    f"Tile {tile_num_actual}\nNo data",
                    ha="center",
                    va="center",
                )
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            # --- Model Prediction ---
            pred_scaled = lstm(X_test).detach().cpu().numpy()[:, future]
            true_scaled = y_test_scaled.numpy()[:, future]

            # --- Time Axis Preparation ---
            start_of_preds_idx = num_in - 1 + future
            time_pred = time[start_of_preds_idx : start_of_preds_idx + len(pred_scaled)]
            time_pred_mpl = mdates.date2num(time_pred)

            # --- Plotting SCALED DATA DIRECTLY ---
            ax.plot(
                time_pred_mpl, true_scaled, color="black", label="Observed (Scaled)"
            )
            ax.plot(
                time_pred_mpl,
                pred_scaled,
                color="red",
                linestyle="--",
                label="Predicted (Scaled)",
            )

            ax.set_title(f"Tile {tile_num_actual}")
            ax.set_ylabel(r"Scaled $d(\mathrm{Flux})/dt$")
            ax.grid(True, linestyle="--")
            ax.legend(loc="upper left", fontsize=8)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

        fig.suptitle(f"LSTM SCALED Derivative Prediction for AR{test_AR}", fontsize=16)

        if save_fig:
            results_dir = os.path.join(
                path, "SAR_EMERGENCE_RESEARCH", "lstm", "results"
            )
            os.makedirs(results_dir, exist_ok=True)
            save_path = os.path.join(
                results_dir,
                f"AR{test_AR}_{os.path.splitext(os.path.basename(filename))[0]}_SCALED.png",
            )
            plt.savefig(save_path, dpi=300)
            print(f"\nResults saved at: {save_path}")
        plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    eval_AR_emergence_with_plots(
        device=device,
        save_fig=True,
        path="../",  # Adjust path to be relative to this script's location
    )
