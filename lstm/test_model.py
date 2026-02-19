from functions import (
    lstm_ready,
    calculate_metrics,
    emergence_indication,
    recalibrate,
    highlight_tile,
    scale_and_combine_data,
    get_params,
    AR_defs,
    load_ar_data,
    RESULTS_PATH,
)
import matplotlib.ticker as mticker
import matplotlib.image as mpimg
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import warnings
import torch
import os
import json
from collections import OrderedDict
from datetime import datetime


from functions import VanillaLSTM
from functions import LSTM

warnings.filterwarnings("ignore")


def initialize_lstm(
    model_class_type, inputs, hidden_size, num_layers, num_pred, filepath, device
):
    input_size = np.shape(inputs)[1]
    ModelClass = VanillaLSTM if model_class_type == "VanillaLSTM" else LSTM

    lstm = ModelClass(input_size, hidden_size, num_layers, num_pred).to(device)
    saved_state_dict = torch.load(filepath, map_location=device)
    new_state_dict = OrderedDict()
    for k, v in saved_state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v
    lstm.load_state_dict(new_state_dict)
    lstm.eval()
    return lstm


def eval_AR_emergence_with_plots(
    device,
    test_AR,
    save_fig,
    path,
    num_pred=12,
    rid_of_top=4,
    num_in=110,
    num_layers=None,
    hidden_size=None,
):
    filename = input("Enter Model Name from results path: ")
    filepath = RESULTS_PATH + filename
    (
        model_type,
        num_layers,
        hidden_size,
        learning_rate,
        dropout,
    ) = get_params(filename)

    # Load scales from scales.json (computed from 41 training ARs)
    scales_path = os.path.join(os.path.dirname(__file__), "scales.json")
    with open(scales_path, "r") as f:
        scales = json.load(f)
    m_scale = tuple(scales["m_scale"])
    flux_scale = tuple(scales["flux_scale"])
    cont_int_scale = tuple(scales["cont_int_scale"])
    num_in = scales["num_in"]
    rid_of_top = scales["rid_of_top"]

    all_emergences = []
    rows = ["AR11698", "AR11726", "AR13165", "AR13179", "AR13183"]
    AR_pred = []
    for test_AR in [11698, 11726, 13165, 13179, 13183]:
        AR_emergences = []
        (
            before_plot,
            _,  # num_in from AR_defs unused; LSTM uses training num_in
            _,
            _,
            starting_tile,
            window_start,
            end,
            start,
        ) = AR_defs(test_AR)
        if not before_plot:
            return
        # Load and scale AR data
        size = 9
        maps, flux, cont_int, time = load_ar_data(test_AR, size, rid_of_top)
        inputs, mag_flux = scale_and_combine_data(
            maps, flux, cont_int, m_scale, flux_scale, cont_int_scale
        )

        lstm = initialize_lstm(
            model_type, inputs, hidden_size, num_layers, num_pred, filepath, device
        )

        # Prepare figure
        fig = plt.figure(figsize=(12, 10))
        main_gs = gridspec.GridSpec(4, 2, figure=fig)

        future = 11
        all_metrics = []
        threshold = 0.01
        sust_time = 4
        window_end = window_start + 72
        allNeeded = []
        firstTimePred = float("inf")
        firstTimeTrue = float("inf")
        lineStylesTrue = set()
        lineStylesPred = set()
        minIpred = set()
        minItrue = set()
        for i in range(7):
            print("Tile {}".format(starting_tile + i + 1))

            X_test, y_test, _ = lstm_ready(
                1 + i, size, inputs, mag_flux, num_in, num_pred
            )
            X_test = X_test.to(device)

            all_predictions = lstm(X_test)
            pred = all_predictions[:, future].detach().cpu().numpy()
            true = y_test[:, future].numpy()
            allNeeded.append([pred, true])

            # Index before prediction starts
            last_known_idx = np.shape(mag_flux[1 + i, :])[0] - np.shape(true)[0] - 1

            mag_before_pred = mag_flux[
                1 + i, last_known_idx - before_plot : last_known_idx
            ]

            d_true = np.gradient(np.concatenate((mag_before_pred, true)))
            indicator_true = emergence_indication(d_true, threshold, sust_time)

            for idx, indic in enumerate(indicator_true):
                if indic == 1:
                    if idx < firstTimeTrue:
                        lineStylesTrue = {i}
                        minItrue = {i}
                        firstTimeTrue = idx
                    elif idx == firstTimeTrue:
                        lineStylesTrue.add(i)
                        minItrue.add(i)
                    break

            d_pred = np.gradient(pred)
            indicator_pred = emergence_indication(d_pred, threshold, sust_time)
            for idx, indic in enumerate(indicator_pred):
                if indic == 1:
                    if idx < firstTimePred:
                        lineStylesPred = {i}
                        minIpred = {i}
                        firstTimePred = idx
                    elif idx == firstTimePred:
                        lineStylesPred.add(i)
                        minIpred.add(i)
                    break
        print("FIRST PREDICTION", firstTimePred)
        firstTimePred -= 12
        for i in range(7):
            pred, true = allNeeded[i]
            last_known_idx = (
                np.shape(mag_flux[1 + i, :])[0] - np.shape(true)[0] - 1
            )  # the index in the timeline before we start predicting
            pred = recalibrate(pred, mag_flux[1 + i, last_known_idx])
            pred = pred[start : len(pred) + end]
            true = true[start : len(true) + end]

            mag_before_pred = mag_flux[
                1 + i, last_known_idx - before_plot : last_known_idx
            ]

            time_cut = time[
                last_known_idx - before_plot : last_known_idx + np.shape(pred)[0]
            ]

            time_cut_mpl = mdates.date2num(time_cut)
            print("WINDOW START:", time_cut[window_start])
            print("WINDOW END:", time_cut[window_end])
            nan_array = np.full(mag_before_pred.shape, np.nan)
            zeros_array = np.full(mag_before_pred.shape, 0)

            true_emergence = None
            pred_emergence = None

            x_time_true = time_cut_mpl[firstTimeTrue]
            x_time_pred = time_cut_mpl[firstTimePred + len(mag_before_pred)]

            ### Plot
            gs = gridspec.GridSpecFromSubplotSpec(
                3, 1, subplot_spec=main_gs[i], height_ratios=[4, 1, 1], hspace=0.05
            )

            # Main plot
            ax0 = plt.subplot(gs[0])
            ax0.plot(
                time_cut_mpl,
                np.concatenate((nan_array, pred)),
                color="red",
                label="Predicted",
            )
            ax0.plot(
                time_cut_mpl,
                np.concatenate((mag_before_pred, true)),
                color="black",
                label="Observed",
            )
            ax0.axvspan(
                time_cut_mpl[window_start],
                time_cut_mpl[window_end],
                color="yellow",
                alpha=0.25,
            )
            if i == 0:
                ax0.legend(loc="upper left")
            ax0.set_ylabel(
                f"Tile {starting_tile + i + 2}"
            )  # Title for each plot (optional)
            ax0.set_ylim([-0.1, 1.1])
            ax0.grid(
                True, which="both", axis="both", linestyle="--", linewidth=0.5
            )  # Enable the grid explicitly
            ax0.tick_params(
                axis="x", which="both", labelbottom=False
            )  # Hide x-axis tick labels
            ax0.xaxis_date()  # Assuming ax0 should interpret x-axis values as dates
            ax0.xaxis.set_major_locator(
                mdates.DayLocator()
            )  # Set major ticks to show once per day
            ax0.xaxis.set_major_formatter(
                mdates.DateFormatter("%Y-%m-%d")
            )  # Format the date
            ax0.axvline(
                x=x_time_pred,  # convert datetime to Matplotlib's internal format
                color="blue",
                linestyle="-" if i in lineStylesPred else "--",
                linewidth=1.2,
                label="First Warning",
            )
            ax0.axvline(
                x=x_time_true,  # convert datetime to Matplotlib's internal format
                color="red",
                linestyle="-" if i in lineStylesTrue else "--",
                linewidth=1.2,
                label="First Emergence",
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
            plt.xticks(rotation=45, ha="right")

            # Subplot d_true
            ax1 = plt.subplot(gs[1])
            d_true = np.gradient(np.concatenate((mag_before_pred, true)))
            indicator_true = emergence_indication(d_true, threshold, sust_time)
            for idx, indic in enumerate(indicator_true):
                if indic == 1:
                    break
            first = True
            for j in range(len(d_true) - 1):  # Now, plot using time_cut_mpl as x-values
                current_color = "g" if indicator_true[j] == 0 else "r"
                if current_color == "r" and first:
                    readable_time = [
                        mdates.num2date(time) for time in time_cut_mpl[j : j + 2]
                    ]
                    first = False
                    print(
                        "Observed First Emergence Time: {}".format(
                            readable_time[1].strftime("%Y-%m-%d %H:%M:%S")
                        )
                    )
                    true_emergence: datetime = readable_time[1]
                ax1.plot(
                    time_cut_mpl[j : j + 2], d_true[j : j + 2], color=current_color
                )  # Use time_cut_mpl for x-values
            ax1.xaxis_date()  # Interpret x-axis values as dates
            ax1.xaxis.set_major_locator(
                mdates.DayLocator()
            )  # Set major ticks to show once per day
            ax1.xaxis.set_major_formatter(
                mdates.DateFormatter("%Y-%m-%d")
            )  # Format the date
            ax1.set_xticklabels([])
            ax1.set_ylim([-0.05, 0.05])
            ax1.set_yticks([0])
            ax1.grid(True, which="both", axis="both", linestyle="--", linewidth=0.5)
            ax1.set_ylabel(r"$\frac{d Obs}{dt}$")
            ax1.axvline(
                x=x_time_pred,  # convert datetime to Matplotlib's internal format
                color="blue",
                linestyle="-" if i in lineStylesPred else "--",
                linewidth=1.2,
                label="First Warning",
            )
            ax1.axvline(
                x=x_time_true,  # convert datetime to Matplotlib's internal format
                color="red",
                linestyle="-" if i in lineStylesTrue else "--",
                linewidth=1.2,
                label="First Emergence",
            )
            # Subplot d_pred
            ax2 = plt.subplot(gs[2])
            d_pred = np.gradient(pred)
            dd_pred = np.gradient(d_pred)
            indicator_pred = emergence_indication(d_pred, threshold, sust_time)
            dd_pred = np.concatenate((zeros_array, dd_pred))
            d_pred = np.concatenate((zeros_array, d_pred))
            indicator_pred = np.concatenate((nan_array, indicator_pred))
            time_cut_mpl = mdates.date2num(
                time_cut
            )  # Convert datetime objects to Matplotlib dates
            first = True
            for k in range(
                len(dd_pred) - 1
            ):  # Now, plot using time_cut_mpl as x-values
                current_color = (
                    "g"
                    if indicator_pred[k] == 0
                    else "r"
                    if indicator_pred[k] == 1
                    else "grey"
                )
                if current_color == "r" and first:
                    readable_time = [
                        mdates.num2date(time) for time in time_cut_mpl[k : k + 2]
                    ]
                    first = False
                    print(
                        "Predicted First Emergence Time: {}".format(
                            readable_time[1].strftime("%Y-%m-%d %H:%M:%S")
                        )
                    )
                    pred_emergence: datetime = readable_time[1]
                alph = 1 if indicator_pred[k] in [0, 1] else 0
                ax2.plot(
                    time_cut_mpl[k : k + 2],
                    d_pred[k : k + 2],
                    color=current_color,
                    alpha=alph,
                )  # Use time_cut_mpl for x-values
            ax2.xaxis_date()  # Tell Matplotlib to interpret the x-axis values as dates
            ax2.xaxis.set_major_locator(
                mdates.DayLocator()
            )  # Set major ticks to show once per day
            ax2.xaxis.set_major_formatter(
                mdates.DateFormatter("%d/%m/%y")
            )  # Format the date
            ax2.tick_params(
                axis="x", which="major", labelsize=9
            )  # Adjust 'labelsize' as needed
            ax2.set_ylim([-0.05, 0.05])
            ax2.set_yticks([0])  # Set the y-axis to only have a tick at 0
            ax2.grid(True, which="both", axis="both", linestyle="--", linewidth=0.5)
            ax2.set_ylabel(r"$\frac{d Pred}{dt}$")
            ax2.axvline(
                x=x_time_true,  # convert datetime to Matplotlib's internal format
                color="red",
                linestyle="-" if i in lineStylesTrue else "--",
                linewidth=1.2,
                label="First Emergence",
            )
            ax2.axvline(
                x=x_time_pred,  # convert datetime to Matplotlib's internal format
                color="blue",
                linestyle="-" if i in lineStylesPred else "--",
                linewidth=1.2,
                label="First Warning",
            )

            # Evaluation metrics
            metrics = calculate_metrics(
                true[window_start:window_end], pred[window_start:window_end]
            )
            all_metrics.append(metrics)
            # print(f"MAE: {metrics[0]}")
            # print(f"MSE: {metrics[1]}")
            print(f"RMSE: {metrics[2]}")
            # print(f"RMSLE: {metrics[3]}")
            # print(f"R-squared: {metrics[4]}")
            to_append = f"Tile {starting_tile + i + 1} \n"
            if pred_emergence is None and true_emergence is None:
                to_append += "Quiet"
            elif pred_emergence and true_emergence is None:
                to_append += "ILAP"
            elif pred_emergence is None and true_emergence:
                to_append += "NO PRED"
            else:
                diff = pred_emergence - true_emergence
                hours = 12 - (diff.days * 24 * 60 + (diff.seconds / 60)) // 60
                to_append += f"{hours:.0f}h Alarm"
            AR_emergences.append(to_append)

        diff = mdates.num2date(x_time_true) - mdates.num2date(x_time_pred)
        hours = (diff.days * 24 * 60 + (diff.seconds / 60)) // 60
        AR_pred.append(hours)
        all_emergences.append(AR_emergences)
        # Last subplot with mag flux
        plt.subplot(4, 2, 8)
        gs_last = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=main_gs[7], wspace=0.05
        )  # Create a GridSpec for the last subplot area with 1 row and 2 columns
        ax_image1 = plt.subplot(gs_last[0, 0])
        ax_image2 = plt.subplot(gs_last[0, 1])
        img1 = mpimg.imread(f"lstm/imgs/AR{test_AR}s.png")
        img2 = mpimg.imread(f"lstm/imgs/AR{test_AR}e.png")
        ax_image1.imshow(
            img1,
            origin="lower",
            extent=[0, 9, 0, 9],  # map image to a 9x9 domain
            interpolation="nearest",
            zorder=0,
        )
        ax_image2.imshow(
            img2,
            origin="lower",
            extent=[0, 9, 0, 9],  # map image to a 9x9 domain
            interpolation="nearest",
            zorder=0,
        )
        ax_image1.set_title("Window Start", fontsize=10)
        ax_image2.set_title("Window End", fontsize=10)

        # --- make both axes a 9x9 square and draw tile gridlines ---
        for ax in (ax_image1, ax_image2):
            ax.set_xlim(0, 9)
            ax.set_ylim(0, 9)
            ax.set_box_aspect(1)
            ax.set_aspect("equal", adjustable="box")
            ax.set_anchor("C")

            # grid at every integer line (0..9)
            ax.xaxis.set_major_locator(mticker.FixedLocator(np.arange(0, 10, 1)))
            ax.yaxis.set_major_locator(mticker.FixedLocator(np.arange(0, 10, 1)))
            ax.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.4)

            ax.tick_params(
                axis="both",
                which="both",
                bottom=False,
                top=False,
                left=False,
                right=False,
                labelleft=False,
                labelbottom=False,
            )

        for tile_num in range(starting_tile, starting_tile + 7):
            highlight_tile(ax_image1, tile_num + 2)
            highlight_tile(ax_image2, tile_num + 2)
        # Calculate average metrics across tiles
        all_metrics_np = np.array(all_metrics)
        means = np.mean(all_metrics_np, axis=0)
        mae_string = r"Average metrics for all tiles plotted:  $\mathrm{{MAE}} = {}$,  $\mathrm{{MSE}} = {}$,  $\mathrm{{RMSE}} = {}$,  $\mathrm{{RMSLE}} = {}$,  $R^2 = {}$".format(
            round(means[0], 3),
            round(means[1], 3),
            round(means[2], 3),
            round(means[3], 3),
            round(means[4], 3),
        )
        fig.text(0.5, 0.02, mae_string, ha="center", va="bottom", fontsize=10)

        plt.tight_layout()  # Adjusts subplot parameters for better layout
        plt.subplots_adjust(top=0.96, bottom=0.075)  # Adjust top spacing
        plt.suptitle(
            "LSTM Results for AR{} (TW = {}h, RoT = {}, In = {}h)".format(
                test_AR, num_pred, rid_of_top, num_in
            ),
            y=0.99,
        )

        save_path = RESULTS_PATH + "/AR{}.png".format(test_AR)
        plt.savefig(save_path)
        plt.close("all")
    print(AR_pred)
    sb = plt.subplot()
    sb.axis("off")
    tbl = sb.table(cellText=all_emergences, rowLabels=rows, loc="center")
    tbl.scale(1, 1.8)

    fig = sb.figure
    fig.savefig(
        RESULTS_PATH + "/table.png", bbox_inches="tight", pad_inches=0, dpi=300
    )  # add transparent=True if desired
    plt.close(fig)

    print("Results saved at: " + save_path)


def eval_AR_emergence(
    device,
    test_AR,
    save_fig,
    path,
    state_dict=None,
    num_pred=None,
    rid_of_top=None,
    num_in=None,
    num_layers=None,
    hidden_size=None,
    n_epochs=None,
    learning_rate=None,
    dropout=None,
    batch_size=None,
):
    filename = None
    if not state_dict:
        (
            num_pred,
            rid_of_top,
            num_in,
            num_layers,
            hidden_size,
            n_epochs,
            learning_rate,
            dropout,
            filename,
        ) = get_params(path)
    print(
        f"Extracted from filename: Time Window: {num_pred}, Rid of Top: {rid_of_top}, Number of Inputs: {num_in}, Number of Layers: {num_layers}, Hidden Size: {hidden_size}, Number of Epochs: {n_epochs}, Learning Rate: {learning_rate}"
    )  # Print extracted values for confirmation

    before_plot, _, _, _, starting_tile, window_start, end, start = AR_defs(test_AR)
    if not before_plot:
        return

    # Load scales from scales.json
    scales_path = os.path.join(os.path.dirname(__file__), "scales.json")
    with open(scales_path, "r") as f:
        scales = json.load(f)
    m_scale = tuple(scales["m_scale"])
    flux_scale = tuple(scales["flux_scale"])
    cont_int_scale = tuple(scales["cont_int_scale"])
    num_in = scales["num_in"]
    rid_of_top = scales["rid_of_top"]

    size = 9
    maps, flux, cont_int, time = load_ar_data(test_AR, size, rid_of_top)
    inputs, mag_flux = scale_and_combine_data(
        maps, flux, cont_int, m_scale, flux_scale, cont_int_scale
    )
    lstm = initialize_lstm(
        inputs, hidden_size, num_layers, num_pred, state_dict, filename, device
    )

    # Assuming prediction, y_test_tensors, ARs, learning_rate, and n_epochs are already defined
    # Loop to create 8 plots
    future = 11
    all_metrics = []
    window_end = window_start + 72

    for i in range(7):
        # print("Tile {}".format(1 + i))

        ### Validation
        X_test, y_test, _ = lstm_ready(
            1 + i, size, inputs, mag_flux, num_in, num_pred
        )  # ,min_p,max_p,min_i,max_i)
        X_test = X_test.to(device)

        all_predictions = lstm(X_test)
        pred = all_predictions[:, future].detach().cpu().numpy()
        true = y_test[:, future].numpy()
        last_known_idx = (
            np.shape(mag_flux[1 + i, :])[0] - np.shape(true)[0] - 1
        )  # the index in the timeline before we start predicting
        pred = recalibrate(pred, mag_flux[1 + i, last_known_idx])
        # Evaluation metrics
        metrics = calculate_metrics(
            true[window_start:window_end], pred[window_start:window_end]
        )
        all_metrics.append(metrics)

    # Print the metrics at the bottom
    all_metrics_np = np.array(
        all_metrics
    )  # Convert all_metrics to a NumPy array for easier manipulation
    means = np.mean(
        all_metrics_np, axis=0
    )  # Calculate the mean and standard deviation for each metric across the 7 runs
    return means[2]


if __name__ == "__main__":
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # Define the device (either 'cuda' for GPU or 'cpu' for CPU)
    print("Runs on: {}".format(device), " / Using", torch.cuda.device_count(), "GPUs!")

    eval_AR_emergence_with_plots(
        device, [11698, 11726, 13165, 13179, 13183], True, "../"
    )
