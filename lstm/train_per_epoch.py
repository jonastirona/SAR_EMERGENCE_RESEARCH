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
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from ray import tune

warnings.filterwarnings("ignore")


def main(
    device,
    config,
    num_pred,
    rid_of_top,
    num_in,
    num_layers,
    hidden_size,
    n_epochs,
    learning_rate,
    dropout,
):
    path = "/mmfs1/project/mx6/ebd/"

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
    # Define the path for the results file

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
        intensities = np.load(
            path
            + "SAR_EMERGENCE_RESEARCH/data/AR{}/mean_int{}_flat.npz".format(AR, AR),
            allow_pickle=True,
        )
        power_maps23 = pm_and_int["arr_0"]
        power_maps34 = pm_and_int["arr_1"]
        power_maps45 = pm_and_int["arr_2"]
        power_maps56 = pm_and_int["arr_3"]
        mag_flux = mag_flux["arr_0"]
        intensities = intensities["arr_0"]
        # Trim array to get rid of top and bottom 0 tiles
        power_maps23 = power_maps23[rid_of_top * size : -rid_of_top * size, :]
        power_maps34 = power_maps34[rid_of_top * size : -rid_of_top * size, :]
        power_maps45 = power_maps45[rid_of_top * size : -rid_of_top * size, :]
        power_maps56 = power_maps56[rid_of_top * size : -rid_of_top * size, :]
        mag_flux = mag_flux[rid_of_top * size : -rid_of_top * size, :]
        mag_flux[np.isnan(mag_flux)] = 0
        intensities = intensities[rid_of_top * size : -rid_of_top * size, :]
        intensities[np.isnan(intensities)] = 0
        # stack inputs and normalize
        stacked_maps = np.stack(
            [power_maps23, power_maps34, power_maps45, power_maps56], axis=1
        )
        stacked_maps[np.isnan(stacked_maps)] = 0
        min_p = np.min(stacked_maps)
        max_p = np.max(stacked_maps)
        min_m = np.min(mag_flux)
        max_m = np.max(mag_flux)
        min_i = np.min(intensities)
        max_i = np.max(intensities)
        stacked_maps = min_max_scaling(stacked_maps, min_p, max_p)
        mag_flux = min_max_scaling(mag_flux, min_m, max_m)
        intensities = min_max_scaling(intensities, min_i, max_i)
        # Reshape mag_flux to have an extra dimension and then put it with pmaps
        int_reshaped = np.expand_dims(intensities, axis=1)
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

    # 1) Build your train set
    X_train_list, y_train_list = [], []
    # 2) Build your test set
    X_test_list, y_test_list = [], []

    for AR_ in range(len(ARs)):
        pm_int = all_inputs[:, :, :, AR_]
        flux = all_flux[:, :, AR_]
        for tile in range(tiles):
            # train slice
            X_tr, y_tr = lstm_ready(tile, size, pm_int, flux, num_in, num_pred)
            X_tr = torch.reshape(X_tr, (X_tr.shape[0], num_in, X_tr.shape[2]))

            X_train_list.append(X_tr)
            y_train_list.append(y_tr)

            # test slice (you already use tiles//2 for your test)
            X_te, y_te = lstm_ready(tiles // 2, size, pm_int, flux, num_in, num_pred)
            X_te = torch.reshape(X_te, (X_te.shape[0], num_in, X_te.shape[2]))
            X_test_list.append(X_te)
            y_test_list.append(y_te)
    # concatenate
    X_train_all = torch.cat(X_train_list, dim=0).to(device)
    y_train_all = torch.cat(y_train_list, dim=0).to(device)
    X_test_all = torch.cat(X_test_list, dim=0).to(device)
    y_test_all = torch.cat(y_test_list, dim=0).to(device)

    # DataLoaders
    batch_size = 64
    train_loader = DataLoader(
        TensorDataset(X_train_all, y_train_all), batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(X_test_all, y_test_all), batch_size=batch_size, shuffle=False
    )
    # Iterate over ARs and tiles, writing results to the same file
    optimiser = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
    scheduler = StepLR(optimiser, step_size=n_epochs // 10, gamma=0.9)
    for epoch in range(n_epochs):
        lstm.train()
        train_loss = []
        for x, y in train_loader:
            # Shuffle
            indices = torch.randperm(int(np.shape(x)[0]))
            X_train = x[indices]
            y_train = y[indices]

            outputs = lstm.forward(X_train)  # Forward pass
            optimiser.zero_grad()  # Calculate the gradient, manually setting to 0
            loss = loss_fn(outputs, y_train)
            loss.backward()  # Calculates the loss of the loss function
            torch.nn.utils.clip_grad_norm_(lstm.parameters(), max_norm=1.0)
            train_loss.append(loss.item())
            optimiser.step()  # Improve from loss, i.e., backprop

        # Test loss
        lstm.eval()
        test_losses = []
        with (
            torch.no_grad()
        ):  # Turn off gradients for validation, saves memory and computations
            for x, y in test_loader:
                test_preds = lstm(x)
                test_loss = loss_fn(test_preds, y)
                test_losses.append(test_loss.item())
        # avg_train_loss = float(np.mean(train_loss))
        # avg_test_loss = float(np.mean(test_losses))

        learning_rate = scheduler.get_last_lr()[0]
        scheduler.step()


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
    config["time_window"] = 12
    # Save model to wandb as artifact
    model_artifact = wandb.Artifact(
        name=f"transformer_t{config['time_window']}_r{config['rid_of_top']}_i{config['num_in']}_n{config['num_layers']}_e{config['n_epochs']}_l{config['learning_rate']:.5f}_d{config['dropout']:.1f}",
        type="model_n_results",
        description="LSTM Model and Results",
        metadata={
            "learning_rate": config["learning_rate"],
            "epochs": config["n_epochs"],
            "num_layers": config["num_layers"],
            "dropout": config["dropout"],
            "time_window": config["time_window"],
            "rid_of_top": config["rid_of_top"],
            "num_in": config["num_in"],
            "model_type": "LSTM",
        },
    )
    model_artifact.add_file(model_path, name=model_name)
    wandb.log_artifact(model_artifact)

    return lstm.state_dict()


if __name__ == "__main__":
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
    except ValueError as e:
        print("Error: Please ensure that all arguments are numbers.")
        sys.exit(1)

    start_time = time.time()
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # Define the device (either 'cuda' for GPU or 'cpu' for CPU)
    print("Runs on: {}".format(device))
    print("Using", torch.cuda.device_count(), "GPUs!")

    # Login to wandb first
    # print("2")
    # wandb.login(key=wandb_api_key, relogin=True, host="https://api.wandb.ai")
    # print("1")
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    wandb_entity = os.environ.get("WANDB_ENTITY")
    wandb_project = "sar"

    config = {
        "num_pred": num_pred,
        "num_in": num_in,
        "num_layers": num_layers,  # Fixed to 3 layers
        "dropout": dropout,  # Set dropout to 0.3
        "n_epochs": n_epochs,
        "time_window": 12,
        "rid_of_top": rid_of_top,
        "learning_rate": learning_rate,  # Set learning rate to 0.001
        # batch_size will be varied in the experiment
    }
    # Initialize wandb run (one run for the sweep, each trial will be a group)
    wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        config=config,
        name="LSTM_" + str(config),
        notes=f"Grid search comparing transformer performance across batch sizes (64, 128, 256, 512) with constant learning rate {learning_rate} , dropout {dropout}, fixed attention head count (4), and {num_layers} layers",
    )

    main(
        device,
        config,
        num_pred,
        rid_of_top,
        num_in,
        num_layers,
        hidden_size,
        n_epochs,
        learning_rate,
        dropout,
    )
    end_time = time.time()
    print("Elapsed time: {} minutes".format((end_time - start_time) / 60))
