from functions import (
    lstm_ready,
    calculate_metrics,
    emergence_indication,
    smooth_with_numpy,
    recalibrate,
    add_grid_lines,
    highlight_tile,
    process_data,
    get_params,
    AR_defs,
    load_ar_data,
    RESULTS_PATH,
)
from functions import VanillaLSTM
from functions import LSTM

import matplotlib.patches as patches 
import matplotlib.image as mpimg
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import warnings
import torch
import os
from collections import OrderedDict


warnings.filterwarnings("ignore")


def initialize_lstm(
    model_class_type,  # New argument to determine class
    inputs,
    hidden_size,
    num_layers,
    num_pred,
    filepath,
    device,
):
    input_size = np.shape(inputs)[1]

    # Select the class based on the type string

    if model_class_type == "VanillaLSTM":
        print(model_class_type, "vanilla")
        ModelClass = VanillaLSTM
    else:
        print(model_class_type, "lstm")
        ModelClass = LSTM

    # Initialize the selected LSTM and move it to GPU
    lstm = ModelClass(input_size, hidden_size, num_layers, num_pred).to(device)

    saved_state_dict = torch.load(filepath, map_location=device)
    new_state_dict = OrderedDict()
    for k, v in saved_state_dict.items():
        name = k[7:] if k.startswith("module.") else k 
        new_state_dict[name] = v
    lstm.load_state_dict(new_state_dict)
    lstm.eval()  # Set the model to evaluation model
    return lstm


def eval_AR_emergence_with_plots(
    device,
    test_ARs,
    save_fig,
    base_path,
    model_configs,  
):
    # 1. Load All Models and their specific parameters
    loaded_models = []

    for graphName, filename in model_configs:
        filepath = os.path.join(RESULTS_PATH, filename)
        (
            model_type,
            num_layers,
            hidden_size,
            learning_rate,
            dropout,
        ) = get_params(filename)

        loaded_models.append(
            {
                "type": model_type,
                "graphName": graphName,
                "filepath": filepath,
                "filename": filename,
                "params": {
                    "num_pred": 12,
                    "rid_of_top": 4,
                    "num_in": 110,
                    "num_layers": num_layers,
                    "hidden_size": hidden_size,
                },
                "model": None,  # Will init inside AR loop
            }
        )

    all_emergences = []
    rows = ["AR11698", "AR11726", "AR13165", "AR13179", "AR13183"]
    AR_pred = []

    if not isinstance(test_ARs, list):
        test_ARs = [test_ARs]

    for test_AR in test_ARs:
        AR_emergences = []
        (
            before_plot,
            num_in,
            NOAA_first,
            NOAA_second,
            starting_tile,
            window_start,
            end,
            start,
        ) = AR_defs(test_AR)
        for e in loaded_models:
            e["params"]["num_in"] = num_in

        if not before_plot:
            continue

        # Define the AR information
        size = 9
        rid_of_top = 4

        cont_int_scale = (-12419.59375, 3119.267578125)
        flux_scale = (-78.26012229919434, 490.13057708740234)
        m_scale = (-365079096.0, 118424064.0)

        maps, flux, cont_int, time = load_ar_data(
            test_AR, size, rid_of_top, starting_tile
        )
        inputs, mag_flux = process_data(
            maps, flux, cont_int, m_scale, flux_scale, cont_int_scale
        )

        # Initialize LSTMs now that we have inputs
        for m_data in loaded_models:
            m_data["model"] = initialize_lstm(
                m_data["type"],
                inputs,
                m_data["params"]["hidden_size"],
                m_data["params"]["num_layers"],
                m_data["params"]["num_pred"],
                m_data["filepath"],
                device,
            )

        fig = plt.figure(figsize=(14, 12))
        main_gs = gridspec.GridSpec(4, 2, figure=fig)

        future = 11
        all_metrics = []
        threshold = 0.01
        sust_time = 4
        window_end = window_start + 72

        # State tracking for the PRIMARY model (index 0)
        firstTimePred = float("inf")
        firstTimeTrue = float("inf")
        lineStylesTrue = set()
        lineStylesPred = set()

        # --- 1. PRE-CALCULATION FOR PRIMARY MODEL (Index 0) ---
        for i in range(7):
            primary = loaded_models[-1]
            p_params = primary["params"]

            X_test_p, y_test_p, _ = lstm_ready(
                1 + i, size, inputs, mag_flux, p_params["num_in"], p_params["num_pred"]
            )
            X_test_p = X_test_p.to(device)

            # Inference Primary
            pred_p = primary["model"](X_test_p)[:, future].detach().cpu().numpy()
            true_p = y_test_p[:, future].numpy()

            last_known_idx = np.shape(mag_flux[1 + i, :])[0] - np.shape(true_p)[0] - 1

            # True Emergence Check
            mag_before_pred = mag_flux[
                1 + i, last_known_idx - before_plot : last_known_idx
            ]

            d_true = np.gradient(np.concatenate((mag_before_pred, true_p)))
            indicator_true = emergence_indication(d_true, threshold, sust_time)

            for idx, indic in enumerate(indicator_true):
                if indic == 1:
                    if idx < firstTimeTrue:
                        lineStylesTrue = {i}
                        firstTimeTrue = idx
                    elif idx == firstTimeTrue:
                        lineStylesTrue.add(i)
                    break

            # Predicted Emergence Check (Primary Only)
            d_pred_p = np.gradient(pred_p)
            indicator_pred_p = emergence_indication(d_pred_p, threshold, sust_time)

            for idx, indic in enumerate(indicator_pred_p):
                if indic == 1:
                    if idx < firstTimePred:
                        lineStylesPred = {i}
                        firstTimePred = idx
                    elif idx == firstTimePred:
                        lineStylesPred.add(i)
                    break
        print(firstTimePred)
        firstTimePred -= 12

        # --- 2. PLOTTING LOOP ---
        for i in range(7):
            print()
            print("Tile {}".format(starting_tile + i + 1))

            # Use Primary Model for timeline setup
            primary = loaded_models[0]
            X_test_p, y_test_p, _ = lstm_ready(
                1 + i,
                size,
                inputs,
                mag_flux,
                primary["params"]["num_in"],
                primary["params"]["num_pred"],
            )
            true = y_test_p[:, future].numpy()

            last_known_idx = np.shape(mag_flux[1 + i, :])[0] - np.shape(true)[0] - 1

            true = true[start : len(true) + end]

            mag_before_pred = mag_flux[
                1 + i, last_known_idx - before_plot : last_known_idx
            ]

            time_cut = time[
                last_known_idx - before_plot : last_known_idx + np.shape(true)[0]
            ]

            time_cut_mpl = mdates.date2num(time_cut)
            total_len = len(mag_before_pred) + len(true)
            time_cut_mpl = time_cut_mpl[:total_len]

            # Markers
            x_time_true = time_cut_mpl[min(firstTimeTrue, len(time_cut_mpl) - 1)]
            idx_pred_line = firstTimePred + len(mag_before_pred)
            if idx_pred_line >= len(time_cut_mpl):
                idx_pred_line = len(time_cut_mpl) - 1
            x_time_pred = time_cut_mpl[idx_pred_line]

            num_models = len(loaded_models)
            height_ratios = [4, 1, 1]
            gs = gridspec.GridSpecFromSubplotSpec(
                3,
                1,
                subplot_spec=main_gs[i],
                height_ratios=height_ratios,
                hspace=0.05,
            )

            # --- AX0: MAGNITUDE ---
            ax0 = plt.subplot(gs[0])

            # Observed (Black) - Primary Model context
            nan_array = np.full(mag_before_pred.shape, np.nan)
            ax0.plot(
                time_cut_mpl,
                np.concatenate((mag_before_pred, true)),
                color="black",
                label="Observed",
                linewidth=1.5,
            )

            # Overlay Predictions
            colors = ["blue", "orange", "purple", "green"]
            pred_primary_recal = None

            for m_idx, m_data in enumerate(loaded_models):
                m_params = m_data["params"]
                X_test_m, y_test_m, _ = lstm_ready(
                    1 + i,
                    size,
                    inputs,
                    mag_flux,
                    m_params["num_in"],
                    m_params["num_pred"],
                )
                X_test_m = X_test_m.to(device)

                pred_raw = m_data["model"](X_test_m)[:, future].detach().cpu().numpy()

                # Recalibrate
                lk_idx_m = (
                    np.shape(mag_flux[1 + i, :])[0]
                    - np.shape(y_test_m[:, future].numpy())[0]
                    - 1
                )
                pred_recal = recalibrate(pred_raw, mag_flux[1 + i, lk_idx_m])
                pred_recal = pred_recal[start : len(pred_recal) + end]

                if m_idx == len(loaded_models) - 1:
                    pred_primary_recal = pred_recal

                plot_data = np.concatenate((nan_array, pred_recal))

                # Length check
                if len(plot_data) > len(time_cut_mpl):
                    plot_data = plot_data[: len(time_cut_mpl)]
                elif len(plot_data) < len(time_cut_mpl):
                    pad = np.full((len(time_cut_mpl) - len(plot_data),), np.nan)
                    plot_data = np.concatenate((plot_data, pad))

                ax0.plot(
                    time_cut_mpl,
                    plot_data,
                    color=colors[m_idx % len(colors)],
                    label=m_data["graphName"],  # Simple label
                    linestyle="-" if m_idx == len(loaded_models) - 1 else "--",
                    alpha=0.8 if m_idx == len(loaded_models) - 1 else 0.3,
                )

            # Formatting
            ax0.axvspan(
                time_cut_mpl[window_start],
                time_cut_mpl[window_end],
                color="yellow",
                alpha=0.25,
            )
            if i == 0:
                ax0.legend(loc="upper left", fontsize=9)
            ax0.set_ylabel(f"Tile {starting_tile + i + 2}", fontsize=12)
            ax0.set_ylim([-0.1, 1.1])
            ax0.grid(True, which="both", axis="both", linestyle="--", linewidth=0.5)
            ax0.tick_params(axis="x", which="both", labelbottom=False)
            ax0.xaxis_date()
            ax0.xaxis.set_major_locator(mdates.DayLocator())
            ax0.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

            # Primary Warning/Emergence Lines ONLY
            ax0.axvline(
                x=x_time_pred,
                color="blue",
                linestyle="-" if i in lineStylesPred else "--",
                linewidth=1.2,
            )
            ax0.axvline(
                x=x_time_true,
                color="red",
                linestyle="-" if i in lineStylesTrue else "--",
                linewidth=1.2,
            )

            if i == 0:
                ax0.text(
                    x_time_pred,
                    ax0.get_ylim()[1],
                    "First Warning ⚑",
                    color="blue",
                    fontsize=10,
                    ha="right",
                    va="bottom",
                )
                ax0.text(
                    x_time_true,
                    ax0.get_ylim()[1],
                    "⚑ First Emergence",
                    color="red",
                    fontsize=10,
                    ha="left",
                    va="bottom",
                )

            # --- AX1: dObs/dt ---
            ax1 = plt.subplot(gs[1])
            d_true = np.gradient(np.concatenate((mag_before_pred, true)))
            indicator_true = emergence_indication(d_true, threshold, sust_time)

            first = True
            true_emergence_dt = None

            for j in range(len(d_true) - 1):
                col = "black" if indicator_true[j] == 0 else "r"
                if col == "r" and first:
                    first = False
                    if i in lineStylesTrue:
                        readable_time = mdates.num2date(time_cut_mpl[j + 1])
                        true_emergence_dt = readable_time

                ax1.plot(time_cut_mpl[j : j + 2], d_true[j : j + 2], color=col)

            ax1.xaxis_date()
            ax1.xaxis.set_major_locator(mdates.DayLocator())
            ax1.set_xticklabels([])
            ax1.set_ylim([-0.05, 0.05])
            ax1.set_yticks([0])
            ax1.grid(True, linestyle="--", linewidth=0.5)
            ax1.set_ylabel(r"$\frac{d\Phi_m}{dt}$")
            ax1.axvline(
                x=x_time_pred,
                color="blue",
                linestyle="-" if i in lineStylesPred else "--",
                linewidth=1.2,
            )
            ax1.axvline(
                x=x_time_true,
                color="red",
                linestyle="-" if i in lineStylesTrue else "--",
                linewidth=1.2,
            )

            # --- AX2: dPred/dt (All Models) ---
            pred_emergence_dt = None

            for m_idx, m_data in enumerate(loaded_models):
                if m_idx != len(loaded_models) - 1:
                    continue
                ax_pred = plt.subplot(gs[2])
                m_params = m_data["params"]
                X_m, y_m, _ = lstm_ready(
                    1 + i,
                    size,
                    inputs,
                    mag_flux,
                    m_params["num_in"],
                    m_params["num_pred"],
                )
                X_m = X_m.to(device)
                p_raw = m_data["model"](X_m)[:, future].detach().cpu().numpy()
                lk_idx_m = (
                    np.shape(mag_flux[1 + i, :])[0]
                    - np.shape(y_m[:, future].numpy())[0]
                    - 1
                )
                p_rec = recalibrate(p_raw, mag_flux[1 + i, lk_idx_m])
                p_rec = p_rec[start : len(p_rec) + end]

                d_pred = np.gradient(p_rec)
                indicator_pred = emergence_indication(d_pred, threshold, sust_time)

                # Get Pred Time ONLY for Primary
                if m_idx == len(loaded_models) - 1:
                    ind_pred_full = np.concatenate(
                        (np.full(len(mag_before_pred), np.nan), indicator_pred)
                    )
                    for k in range(len(ind_pred_full)):
                        if ind_pred_full[k] == 1:
                            pred_emergence_dt = mdates.num2date(time_cut_mpl[k])
                            break

                d_pred_full = np.concatenate((np.zeros(len(mag_before_pred)), d_pred))
                indicator_pred_full = np.concatenate(
                    (np.zeros(len(mag_before_pred)), indicator_pred)
                )

                for j in range(len(d_pred_full) - 1):
                    col = (
                        colors[m_idx % len(colors)]
                        if indicator_pred_full[j] == 0
                        else "r"
                    )
                    ax_pred.plot(
                        time_cut_mpl[j : j + 2], d_pred_full[j : j + 2], color=col
                    )

                ax_pred.xaxis_date()
                ax_pred.xaxis.set_major_locator(mdates.DayLocator())
                if m_idx < num_models - 1:
                    ax_pred.set_xticklabels([])
                else:
                    ax_pred.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m/%y"))
                    ax_pred.tick_params(axis="x", labelsize=9)

                ax_pred.set_ylim([-0.05, 0.05])
                ax_pred.set_yticks([0])
                ax_pred.grid(True, linestyle="--", linewidth=0.5)
                ax_pred.set_ylabel(r"$\frac{d P}{dt}$ ")
                ax_pred.axvline(
                    x=x_time_true,
                    color="red",
                    linestyle="-" if i in lineStylesTrue else "--",
                    linewidth=1.2,
                )
                ax_pred.axvline(
                    x=x_time_pred,
                    color="blue",
                    linestyle="-" if i in lineStylesPred else "--",
                    linewidth=1.2,
                )

            # Metrics (Primary)
            metrics = calculate_metrics(
                true[window_start:window_end],
                pred_primary_recal[window_start:window_end],
            )
            all_metrics.append(metrics)
            print(f"RMSE (Primary): {metrics[2]}")

            # Table (Primary)
            to_append = f"Tile {starting_tile + i + 1} \n"
            if pred_emergence_dt is None and true_emergence_dt is None:
                to_append += "Quiet"
            elif pred_emergence_dt and true_emergence_dt is None:
                to_append += "ILAP"
            elif pred_emergence_dt is None and true_emergence_dt:
                to_append += "NO PRED"
            else:
                if pred_emergence_dt and true_emergence_dt:
                    diff = pred_emergence_dt - true_emergence_dt
                    hours = 12 - (diff.days * 24 * 60 + (diff.seconds / 60)) // 60
                    to_append += f"{hours:.0f}h Alarm"
                else:
                    to_append += "N/A"
            AR_emergences.append(to_append)

        # Table data cleanup
        diff_final = mdates.num2date(x_time_true) - mdates.num2date(x_time_pred)
        hours_final = (diff_final.days * 24 * 60 + (diff_final.seconds / 60)) // 60
        AR_pred.append(hours_final)
        all_emergences.append(AR_emergences)

        # Images
        gs_last = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=main_gs[7], wspace=0.05
        )
        ax_image1 = plt.subplot(gs_last[0, 0])
        ax_image2 = plt.subplot(gs_last[0, 1])
        ax_image1.axis("off")
        ax_image2.axis("off")
        try:
            img1 = mpimg.imread(f"lstm/imgs/AR{test_AR}s.png")
            img2 = mpimg.imread(f"lstm/imgs/AR{test_AR}e.png")
            for ax, img, title in zip(
                [ax_image1, ax_image2], [img1, img2], ["Window Start", "Window End"]
            ):
                ax.imshow(
                    img,
                    origin="lower",
                    extent=[0, 9, 0, 9],
                    interpolation="nearest",
                    zorder=0,
                )

                # --- draw 9x9 tile grid (1x1 squares from 0..9 in x and y) ---
                for ix in range(9):  # columns
                    for iy in range(9):  # rows
                        rect = patches.Rectangle(
                            (ix, iy),  # bottom-left corner
                            1,
                            1,  # width, height
                            fill=False,
                            linewidth=0.4,
                            edgecolor="white",
                            zorder=1,
                        )
                        ax.add_patch(rect)

                ax.set_title(title, fontsize=12)
                ax.axis("off")
                for tile_num in range(starting_tile, starting_tile + 7):
                    highlight_tile(ax, tile_num + 2)
        except Exception:
            pass

        plt.tight_layout()
        plt.subplots_adjust(top=0.96, bottom=0.075)
        plt.suptitle("Comparison Results for AR{}".format(test_AR), y=0.99, fontsize=15)

        save_path = RESULTS_PATH + "/AR{}_comparison.png".format(test_AR)
        plt.savefig(save_path)
        plt.close("all")
        print("Results saved at: " + save_path)

    sb = plt.subplot()
    sb.axis("off")
    tbl = sb.table(cellText=all_emergences, rowLabels=rows, loc="center")
    tbl.scale(1, 1.8)
    fig = sb.figure
    fig.savefig(
        RESULTS_PATH + "/table_comparison.png",
        bbox_inches="tight",
        pad_inches=0,
        dpi=300,
    )
    plt.close(fig)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models_to_compare = [
        (
            "MagFluxSeq2Seq MSE Loss",
            "../models/LSTM12_r4_i110_n4_h32_e8_lr0.00170074_d0.3.pth",
        ),
        (
            "MagFluxSeq2Seq Hybrid Loss",
            "../models/LSTM12_r4_i110_n4_h64_e10_lr0.00972080_d0.2.pth",
        ),
        (
            "MagFluxLSTM MSE Loss",
            "../models/VanillaLSTM12_r4_i110_n1_h64_e10_lr0.00232000_d0.2.pth",
        ),
        (
            "MagFluxLSTM Hybrid Loss",
            "../models/VanillaLSTM12_r4_i110_n4_h128_e8_lr0.00700000_d0.1.pth",
        ),
    ]

    eval_AR_emergence_with_plots(
        device, [11698, 11726, 13165, 13179, 13183], True, "../", models_to_compare
    )
