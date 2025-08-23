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
    isVanillaLSTM,
)
from sklearn.metrics import mean_squared_error
from datetime import timedelta
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import warnings
import torch
import os
from collections import OrderedDict
from datetime import datetime

if isVanillaLSTM:
    from functions import VanillaLSTM as LSTM
else:
    from functions import LSTM as LSTM

warnings.filterwarnings("ignore")


def initialize_lstm(
    inputs, hidden_size, num_layers, num_pred, state_dict, filename, device
):
    input_size = np.shape(inputs)[1]

    # Initialize the LSTM and move it to GPU
    lstm = LSTM(input_size, hidden_size, num_layers, num_pred).to(device)
    saved_state_dict = state_dict or torch.load(filename, map_location=device)
    new_state_dict = OrderedDict()
    for k, v in saved_state_dict.items():
        name = k[7:] if k.startswith("module.") else k  # remove 'module.' prefix
        new_state_dict[name] = v
    lstm.load_state_dict(new_state_dict)
    lstm.eval()  # Set the model to evaluation model
    return lstm


def eval_AR_emergence_with_plots(
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
        ) = get_params(state_dict, path)
    # print(
    #     f"Extracted from filename: Time Window: {num_pred}, Rid of Top: {rid_of_top}, Number of Inputs: {num_in}, Number of Layers: {num_layers}, Hidden Size: {hidden_size}, Number of Epochs: {n_epochs}, Learning Rate: {learning_rate}"
    # )  # Print extracted values for confirmation
    all_emergences = []
    rows = ["AR11698", "AR11726", "AR13165", "AR13179", "AR13183"]
    for test_AR in [11698, 11726, 13165, 13179, 13183]:
        AR_emergences = []
        before_plot, num_in, NOAA_first, NOAA_second, starting_tile, window_start = (
            AR_defs(test_AR)
        )
        if not before_plot:
            return

        # Define the AR information
        size = 9
        rid_of_top = 4

        inputs, mag_flux, time = process_data(test_AR, size, rid_of_top, starting_tile)
        lstm = initialize_lstm(
            inputs, hidden_size, num_layers, num_pred, state_dict, filename, device
        )

        # Assuming prediction, y_test_tensors, ARs, learning_rate, and n_epochs are already defined
        fig = plt.figure(figsize=(12, 10))  # Adjust the figure size if necessary
        main_gs = gridspec.GridSpec(
            4, 2, figure=fig
        )  # Create a GridSpec with 4 rows and 2 columns

        # Loop to create 8 plots
        future = 11
        all_metrics = []
        threshold = 0.01  # -0.006
        sust_time = 4
        window_end = window_start + 72
        for i in range(7):
            print()
            print("Tile {}".format(starting_tile + i + 1))

            ### Validation
            print("Inputs shape;", inputs.shape, "Mag flux shape:", mag_flux.shape)
            X_test, y_test = lstm_ready(
                1 + i, size, inputs, mag_flux, num_in, num_pred
            )  # ,min_p,max_p,min_i,max_i)
            X_test = X_test.to(device)
            print("x_test shape:", X_test.shape)

            all_predictions = lstm(X_test)
            print("all predictions shape:", all_predictions.shape)
            pred = all_predictions[:, future].detach().cpu().numpy()
            print("pred:", pred.shape)
            print("y_test:", y_test.shape)
            true = y_test[:, future].numpy()

            last_known_idx = (
                np.shape(mag_flux[1 + i, :])[0] - np.shape(true)[0] - 1
            )  # the index in the timeline before we start predicting
            pred = recalibrate(pred, mag_flux[1 + i, last_known_idx])
            first_pred_time = mdates.date2num(
                time[last_known_idx] - timedelta(hours=num_pred)
            )
            mag_before_pred = mag_flux[
                1 + i,
                last_known_idx - before_plot - num_pred - future : last_known_idx
                - num_pred
                - future,
            ]
            time_cut = time[
                last_known_idx - before_plot : last_known_idx + np.shape(pred)[0]
            ]
            time_cut_mpl = mdates.date2num(time_cut)
            nan_array = np.full(mag_before_pred.shape, np.nan)
            zeros_array = np.full(mag_before_pred.shape, 0)

            true_emergence = None
            pred_emergence = None

            ### Plot
            gs = gridspec.GridSpecFromSubplotSpec(
                3, 1, subplot_spec=main_gs[i], height_ratios=[4, 1, 1], hspace=0.05
            )  # Define GridSpec for this iteration
            # true = true[:-before_plot]; pred = pred[:-before_plot]; time_cut_mpl = time_cut_mpl[:-before_plot]

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
            ax0.plot(
                time_cut_mpl,
                smooth_with_numpy(np.concatenate((mag_before_pred, true))),
                color="black",
                alpha=0.25,
                label="Smooth Obs.",
            )
            ax0.axvspan(
                time_cut_mpl[window_start],
                time_cut_mpl[window_end],
                color="yellow",
                alpha=0.25,
            )
            ax0.legend(loc="upper left")
            ax0.set_ylabel(
                f"Tile {starting_tile + i + 1}"
            )  # Title for each plot (optional)
            # ax0.axvline(x=first_pred_time, color='darkturquoise', linestyle='--')
            ### ax0.axvline(x=NOAA_first_record, color='magenta', linestyle='--')  # Adjust color, linestyle, linewidth as needed
            ### ax0.axvline(x=NOAA_second_record, color='darkmagenta', linestyle='--')
            # ax0.legend(['Observed','Predicted', 'Observed (Smooth)', 'First Prediction', r'NOAA $1^{st}$ Record', 'After Emergence'], fontsize = 7)  # Legend for each plot (optional)
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
            plt.xticks(
                rotation=45, ha="right"
            )  # Rotate x-tick labels for better readability

            # Subplot d_true
            ax1 = plt.subplot(gs[1])
            d_true = np.gradient(
                smooth_with_numpy(np.concatenate((mag_before_pred, true)))
            )  # ; d_true = smooth_with_numpy(d_true) # Assuming d_true is your data derivative
            indicator_true = emergence_indication(
                d_true, threshold, sust_time
            )  # emergence_indication2(dd_true) #
            first = True
            for j in range(len(d_true) - 1):  # Now, plot using time_cut_mpl as x-values
                current_color = "g" if indicator_true[j] == 0 else "r"
                if current_color == "r" and first == True:
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
            # ax1.axvline(x=first_pred_time, color='darkturquoise', linestyle='--')
            ### ax1.axvline(x=NOAA_first_record, color='magenta', linestyle='--')  # Adjust color, linestyle, linewidth as needed
            ### ax1.axvline(x=NOAA_second_record, color='darkmagenta', linestyle='--')
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

            # Subplot d_pred
            ax2 = plt.subplot(gs[2])
            d_pred = np.gradient(
                pred
            )  # np.gradient(pred) # Assuming d_pred is your data derivative
            dd_pred = np.gradient(d_pred)
            indicator_pred = emergence_indication(
                d_pred, threshold, sust_time
            )  # emergence_indication2(dd_pred) #
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
                if current_color == "r" and first == True:
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
            # ax2.axvline(x=first_pred_time, color='darkturquoise', linestyle='--')
            ### ax2.axvline(x=NOAA_first_record, color='magenta', linestyle='--')  # Adjust color, linestyle, linewidth as needed
            ### ax2.axvline(x=NOAA_second_record, color='darkmagenta', linestyle='--')
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

        all_emergences.append(AR_emergences)
        # Last subplot with mag flux
        plt.subplot(4, 2, 8)
        gs_last = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=main_gs[7], wspace=0.05
        )  # Create a GridSpec for the last subplot area with 1 row and 2 columns
        ax_image1 = plt.subplot(gs_last[0, 0])  # Plot the first image
        ### ax_image1.imshow(NOAA_first_int_map, cmap='gray')
        add_grid_lines(ax_image1)  # Add grid lines to the first image
        for tile_num in range(starting_tile, starting_tile + 7):
            highlight_tile(
                ax_image1, tile_num
            )  # Loop to highlight tiles from starting_tile to starting_tile + 7
        ax_image1.set_xlabel("{}".format(NOAA_first.strftime("%d/%m/%y %H:%M")))
        ax_image1.tick_params(
            axis="both",
            which="both",
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelleft=False,
            labelbottom=True,
        )
        ax_image1.set_xticks([])
        ax_image1.set_yticks([])
        ax_image2 = plt.subplot(gs_last[0, 1])  # Plot the second image
        ### ax_image2.imshow(NOAA_second_int_map, cmap='gray')
        add_grid_lines(ax_image2)  # Add grid lines to the first image
        for tile_num in range(starting_tile, starting_tile + 7):
            highlight_tile(
                ax_image2, tile_num
            )  # Loop to highlight tiles from starting_tile to starting_tile + 7
        ax_image1.set_title("Magnetic Flux", fontsize=10)
        ax_image2.set_title("Magnetic Flux", fontsize=10)
        ax_image2.set_xlabel("{}".format(NOAA_second.strftime("%d/%m/%y %H:%M")))
        ax_image2.tick_params(
            axis="both",
            which="both",
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelleft=False,
            labelbottom=True,
        )
        ax_image2.set_xticks([])
        ax_image2.set_yticks([])

        # Print the metrics at the bottom
        all_metrics_np = np.array(
            all_metrics
        )  # Convert all_metrics to a NumPy array for easier manipulation
        means = np.mean(
            all_metrics_np, axis=0
        )  # Calculate the mean and standard deviation for each metric across the 7 runs
        stds = np.std(all_metrics_np, axis=0)
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

        score = mean_squared_error(
            np.concatenate((mag_before_pred, true)), np.concatenate((zeros_array, pred))
        )

        save_path = path + "SAR_EMERGENCE_RESEARCH/lstm/results/AR{}_{}.png".format(
            test_AR, os.path.splitext(os.path.basename(filename))[0]
        )
        plt.savefig(save_path)
        plt.close("all")
    sb = plt.subplot()
    sb.axis("off")
    tbl = sb.table(cellText=all_emergences, rowLabels=rows, loc="center")
    tbl.scale(1, 1.8) 

    fig = sb.figure
    fig.savefig(
        path + "SAR_EMERGENCE_RESEARCH/lstm/results/table.png", bbox_inches="tight", pad_inches=0, dpi=300
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
        ) = get_params(state_dict, path)
    print(
        f"Extracted from filename: Time Window: {num_pred}, Rid of Top: {rid_of_top}, Number of Inputs: {num_in}, Number of Layers: {num_layers}, Hidden Size: {hidden_size}, Number of Epochs: {n_epochs}, Learning Rate: {learning_rate}"
    )  # Print extracted values for confirmation

    before_plot, num_in, _, _, starting_tile, window_start = AR_defs(test_AR)
    if not before_plot:
        return

    # Define the AR information
    size = 9
    rid_of_top = 4

    inputs, mag_flux, time = process_data(test_AR, size, rid_of_top, starting_tile)
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
        X_test, y_test = lstm_ready(
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
    stds = np.std(all_metrics_np, axis=0)
    mae_string = r"Average metrics for all tiles plotted:  $\mathrm{{MAE}} = {}$,  $\mathrm{{MSE}} = {}$,  $\mathrm{{RMSE}} = {}$,  $\mathrm{{RMSLE}} = {}$,  $R^2 = {}$".format(
        round(means[0], 3),
        round(means[1], 3),
        round(means[2], 3),
        round(means[3], 3),
        round(means[4], 3),
    )
    # print(mae_string)
    return means[2]


if __name__ == "__main__":
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # Define the device (either 'cuda' for GPU or 'cpu' for CPU)
    print("Runs on: {}".format(device), " / Using", torch.cuda.device_count(), "GPUs!")

    eval_AR_emergence_with_plots(
        device, [11698, 11726, 13165, 13179, 13183], True, "../"
    )
