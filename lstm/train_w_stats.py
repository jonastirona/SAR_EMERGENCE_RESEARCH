import os
import sys
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
from functions import LSTM, lstm_ready, min_max_scaling, training_loop_w_stats
from matplotlib.backends.backend_pdf import PdfPages
import wandb

warnings.filterwarnings("ignore")

path = "/mmfs1/project/mx6/ebd/"
local = True
if local:
    path = "../"


def main(device, wandb_api_key, wandb_entity, wandb_project):
    # Check if the correct number of arguments is provided
    if len(sys.argv) != 9:
        print(
            "Usage: script.py num_pred rid_of_top num_in num_layers hidden_size n_epochs learning_rate dropout"
        )
        sys.exit(1)
    try:  # Extract arguments and convert them to the appropriate types  #python3 train_w_stats.py 12 4 120 3 64 500 0.01 0.1
        num_pred = int(sys.argv[1])
        print("Time Windows:", num_pred)
        rid_of_top = int(sys.argv[2])
        print("Rid of Top:", rid_of_top)
        num_in = int(sys.argv[3])
        print("Number of Inputs:", num_in)
        num_layers = int(sys.argv[4])
        print("Number of Layers:", num_layers)
        hidden_size = int(sys.argv[5])
        print("Hidden Size:", hidden_size)
        n_epochs = int(sys.argv[6])
        print("Number of Epochs:", n_epochs)
        learning_rate = float(sys.argv[7])
        print("Learning Rate:", learning_rate)
        dropout = float(sys.argv[8])
        print("Dropout:", dropout)

        config = {
            "num_pred": num_pred,
            "num_in": num_in,
            "embed_dim": 64,
            "ff_dim": 128,
            "num_layers": num_layers,  # Fixed to 3 layers
            "dropout": dropout,  # Set dropout to 0.3
            "n_epochs": n_epochs,
            "time_window": 12,
            "rid_of_top": rid_of_top,
            "learning_rate": learning_rate,  # Set learning rate to 0.001
            # batch_size will be varied in the experiment
        }
    except ValueError as e:
        print("Error: Please ensure that all arguments are numbers.")
        sys.exit(1)

    # Initialize wandb run (one run for the sweep, each trial will be a group)
    wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        config=config,
        name="batch size search",
        notes=f"Grid search comparing transformer performance across batch sizes (64, 128, 256, 512) with constant learning rate {learning_rate} , dropout {dropout}, fixed attention head count (4), and {num_layers} layers",
    )

    # Now you can use these variables in your script
    ARs = [
        11130,
        11149,
        11158,
        11162,
        11199,
        11327,
        11344,
        11387,
        11393,
        11416,
        11422,
        11455,
        11619,
        11640,
        11660,
        11678,
        11682,
        11765,
        11768,
        11776,
        11916,
        11928,
        12036,
        12051,
        12085,
        12089,
        12144,
        12175,
        12203,
        12257,
        12331,
        12494,
        12659,
        12778,
        12864,
        12877,
        12900,
        12929,
        13004,
        13085,
        13098,
    ]
    flatten = True
    size = 9
    tiles = size**2 - 2 * size * rid_of_top
    test_AR = 13179  # and the secondary will be 13165 and if I fix it, third: 13183
    ARs_ = ARs + [test_AR]
    result_string = []
    # Define the path for the results file
    result_file_path = path + "SAR_EMERGENCE_RESEARCH/lstm/results.txt"

    # Preprocessing
    print("Load data and split in tiles for {} ARs".format(len(ARs)))
    all_inputs = []
    all_flux = []
    for AR in ARs_:
        pm_and_int = np.load(
            path
            + "SAR_EMERGENCE_RESEARCH/data/AR{}/mean_pmdop{}_flat.npz".format(AR, AR),
            allow_pickle=True,
        )
        mag_flux = np.load(
            path
            + "SAR_EMERGENCE_RESEARCH/data/AR{}/mean_mag{}_flat.npz".format(AR, AR),
            allow_pickle=True,
        )
        flux = np.load(
            path
            + "SAR_EMERGENCE_RESEARCH/data/AR{}/mean_int{}_flat.npz".format(AR, AR),
            allow_pickle=True,
        )
        power_maps23 = pm_and_int["arr_0"]
        power_maps34 = pm_and_int["arr_1"]
        power_maps45 = pm_and_int["arr_2"]
        power_maps56 = pm_and_int["arr_3"]
        mag_flux = mag_flux["arr_0"]
        flux = flux["arr_0"]
        # Trim array to get rid of top and bottom 0 tiles
        power_maps23 = power_maps23[rid_of_top * size : -rid_of_top * size, :]
        power_maps34 = power_maps34[rid_of_top * size : -rid_of_top * size, :]
        power_maps45 = power_maps45[rid_of_top * size : -rid_of_top * size, :]
        power_maps56 = power_maps56[rid_of_top * size : -rid_of_top * size, :]
        mag_flux = mag_flux[rid_of_top * size : -rid_of_top * size, :]
        mag_flux[np.isnan(mag_flux)] = 0
        flux = flux[rid_of_top * size : -rid_of_top * size, :]
        flux[np.isnan(flux)] = 0
        # stack inputs and normalize
        stacked_maps = np.stack(
            [power_maps23, power_maps34, power_maps45, power_maps56], axis=1
        )
        stacked_maps[np.isnan(stacked_maps)] = 0
        min_p = np.min(stacked_maps)
        max_p = np.max(stacked_maps)
        min_m = np.min(mag_flux)
        max_m = np.max(mag_flux)
        min_i = np.min(flux)
        max_i = np.max(flux)
        stacked_maps = min_max_scaling(stacked_maps, min_p, max_p)
        mag_flux = min_max_scaling(mag_flux, min_m, max_m)
        flux = min_max_scaling(flux, min_i, max_i)
        # Reshape mag_flux to have an extra dimension and then put it with pmaps
        int_reshaped = np.expand_dims(flux, axis=1)
        pm_and_int = np.concatenate([stacked_maps, int_reshaped], axis=1)
        # append all ARs
        all_inputs.append(pm_and_int)
        all_flux.append(mag_flux)
    all_inputs = np.stack(all_inputs, axis=-1)
    all_flux = np.stack(all_flux, axis=-1)
    input_size = np.shape(all_inputs)[1]

    # Start Training
    lstm = LSTM(input_size, hidden_size, num_layers, num_pred, dropout=dropout).to(
        device
    )
    # if torch.cuda.device_count() > 1: lstm = torch.nn.DataParallel(lstm)
    loss_fn = (
        torch.nn.MSELoss()
    )  # torch.nn.L1Loss() #   # mean-squared error for regression
    # optimiser = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

    os.makedirs(os.path.dirname(result_file_path), exist_ok=True)

    # Open the file once, before the loop
    with open(result_file_path, "w") as file:
        # Dictionary to store all results
        all_training_results = {}

        # Iterate over ARs and tiles, writing results to the same file
        for AR_ in range(len(ARs)):
            pm_and_int = all_inputs[
                :, :, :, AR_
            ]  # change to inputs, its not only power maps
            flux = all_flux[:, :, AR_]
            for tile in range(tiles):
                optimiser = torch.optim.Adam(
                    lstm.parameters(), lr=learning_rate
                )  # WAS MOVED HERE, SEEMS MORE CORRECT
                print("AR{} - Tile: {}".format(ARs[AR_], tile))
                X_train, y_train = lstm_ready(
                    tile, size, pm_and_int, flux, num_in, num_pred
                )
                X_test, y_test = lstm_ready(
                    int(tiles / 2), size, pm_and_int, flux, num_in, num_pred
                )
                # reshaping to rows, timestamps, features
                X_train_final = torch.reshape(
                    X_train, (X_train.shape[0], num_in, X_train.shape[2])
                )
                X_test_final = torch.reshape(
                    X_test, (X_test.shape[0], num_in, X_test.shape[2])
                )
                # Move data to GPU
                X_train_final = X_train_final.to(device)
                y_train = y_train.to(device)
                X_test_final = X_test_final.to(device)
                y_test = y_test.to(device)
                results = training_loop_w_stats(
                    n_epochs=n_epochs,
                    lstm=lstm,
                    optimiser=optimiser,
                    loss_fn=loss_fn,
                    X_train=X_train_final,
                    y_train=y_train,
                    X_test=X_test_final,
                    y_test=y_test,
                )

                # Store results for plotting
                all_training_results[f"AR{ARs[AR_]}_Tile{tile}"] = results

                # Write AR and tile header
                result_string.append(f"\nAR {ARs[AR_]} - Tile {tile}\n")
                result_string.append("Epoch, Train Loss, Test Loss, Learning Rate\n")
                for epoch, train_loss, test_loss, lr in results:
                    result_string.append(
                        f"{epoch}, {train_loss:.5f}, {test_loss:.5f}, {lr:.5f}\n"
                    )
                file.write("".join(result_string))

    # Create PDF with loss curves
    pdf_path = f"t{num_pred}_r{rid_of_top}_i{num_in}_n{num_layers}_h{hidden_size}_e{n_epochs}_l{learning_rate}_d{dropout}_loss_curves"

    ## SAVE LOSS CURVES
    # Create summary plots
    fig = plt.figure(figsize=(15, 12))  # Increased height for new metrics
    gs = plt.GridSpec(3, 2, figure=fig)  # Changed to 3 rows

    # Plot 1: Average losses across all tiles
    ax1 = fig.add_subplot(gs[0, :])
    avg_train_losses = np.mean(
        [
            np.array([r[1] for r in results])
            for results in all_training_results.values()
        ],
        axis=0,
    )
    avg_test_losses = np.mean(
        [
            np.array([r[2] for r in results])
            for results in all_training_results.values()
        ],
        axis=0,
    )
    epochs = range(n_epochs)

    ax1.plot(epochs, avg_train_losses, label="Avg Training Loss")
    ax1.plot(epochs, avg_test_losses, label="Avg Test Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Average Loss")
    ax1.set_title("Average Training and Test Loss Across All Tiles")
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Average learning rate
    ax2 = fig.add_subplot(gs[1, 0])
    avg_lr = np.mean(
        [
            np.array([r[3] for r in results])
            for results in all_training_results.values()
        ],
        axis=0,
    )
    ax2.plot(epochs, avg_lr, label="Avg Learning Rate", color="green")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Learning Rate")
    ax2.set_title("Average Learning Rate Schedule")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.suptitle(
        f"Training Summary - LSTM\nParameters: TW={num_pred}, RoT={rid_of_top}, In={num_in}, Layers={num_layers}, Hidden={hidden_size}",
        fontsize=12,
    )
    wandb.log({pdf_path: fig})
    plt.close(fig)

    print(f"Loss curves saved at: {pdf_path} in wandb")

    # Save the model weights
    model_name = "t{}_r{}_i{}_n{}_h{}_e{}_l{}_d{}.pth".format(
        num_pred,
        rid_of_top,
        num_in,
        num_layers,
        hidden_size,
        n_epochs,
        learning_rate,
        dropout,
    )
    model_path = path + "SAR_EMERGENCE_RESEARCH/lstm/results/{model_name}"
    torch.save(lstm.state_dict(), model_path)

    # Save model to wandb as artifact
    model_artifact = wandb.Artifact(
        name=f"transformer_t{config['time_window']}_r{config['rid_of_top']}_i{config['num_in']}_n{config['num_layers']}_h{config['embed_dim']}_e{config['n_epochs']}_l{config['learning_rate']:.5f}_d{config['dropout']:.1f}",
        type="model_n_results",
        description="LSTM Model and Results",
        metadata={
            "learning_rate": config["learning_rate"],
            "epochs": config["n_epochs"],
            "embed_dim": config["embed_dim"],
            "ff_dim": config["ff_dim"],
            "num_layers": config["num_layers"],
            "dropout": config["dropout"],
            "time_window": config["time_window"],
            "rid_of_top": config["rid_of_top"],
            "num_in": config["num_in"],
            "model_type": "LSTM",
        },
    )
    model_artifact.add_file(model_path, name=model_name)
    model_artifact.add_file(result_file_path, name="results.txt")
    wandb.log_artifact(model_artifact)


if __name__ == "__main__":
    start_time = time.time()
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # Define the device (either 'cuda' for GPU or 'cpu' for CPU)
    print("Runs on: {}".format(device))
    print("Using", torch.cuda.device_count(), "GPUs!")

    wandb_api_key = os.environ.get("WANDB_API_KEY")
    wandb_entity = os.environ.get("WANDB_ENTITY")
    wandb_project = os.environ.get("WANDB_PROJECT")

    # Login to wandb first
    # print("2")
    # wandb.login(key=wandb_api_key, relogin=True, host="https://api.wandb.ai")
    # print("1")

    main(device, wandb_api_key, wandb_entity, wandb_project)
    end_time = time.time()
    print("Elapsed time: {} minutes".format((end_time - start_time) / 60))
