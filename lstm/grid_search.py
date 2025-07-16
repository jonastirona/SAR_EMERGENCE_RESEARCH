# from train_per_epoch import main as train
import torch
from eval import eval_AR_emergence_with_plots as eval
import wandb
import numpy as np
import matplotlib.pyplot as plt
from ray import tune
import ray
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler
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


path = "/mmfs1/project/mx6/ebd/"
# path = "C:/Projects/"


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
    # Now you can use these variables in your script
    print(ray.available_resources())
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
        pm_and_int = all_inputs[:, :, :, AR_]  # shape (time, features, tiles)
        flux = all_flux[:, :, AR_]

        for tile in range(tiles):
            # prepare train split
            X_tr, y_tr = lstm_ready(tile, size, pm_and_int, flux, num_in, num_pred)
            # reshape into (n_samples, num_in, n_features)
            print(X_tr.shape)
            X_tr = X_tr.reshape(X_tr.shape[0], num_in, X_tr.shape[2])
            X_train_list.append(X_tr)
            y_train_list.append(y_tr)

            # prepare test split (use tiles//2 as you had)
            X_te, y_te = lstm_ready(
                tiles // 2, size, pm_and_int, flux, num_in, num_pred
            )
            X_te = X_te.reshape(X_te.shape[0], num_in, X_te.shape[2])
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

        learning_rate = scheduler.get_last_lr()[0]
        scheduler.step()

        scores = []
        for AR in [11698, 11726, 13165, 13179, 13183]:
            score, fig = eval(
                device,
                AR,
                False,
                path,
                lstm.state_dict(),
                config["num_pred"],
                config["rot"],
                config["num_in"],
                config["n_layers"],
                config["hidden_size"],
                config["epochs"],
                config["lr"],
                config["dropout"],
            )
            plt.close(fig)
            scores.append(score)
        val_rmse = float(np.mean(scores))
        print(f"Score: {val_rmse:.8f}")
        tune.report(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(train_loss)),
                "test_loss": float(np.mean(test_losses)),
                "score": val_rmse,
                "lr": learning_rate,
            }
        )

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
    model_path = path + f"SAR_EMERGENCE_RESEARCH/lstm/results/{model_name}"
    torch.save(lstm.state_dict(), model_path)

    return lstm.state_dict()


def objective(config):
    print("Training with", config)
    device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)  # Define the device (either 'cuda' for GPU or 'cpu' for CPU)
    print("Runs on: {}".format(device))
    print("Using", torch.cuda.device_count(), "GPUs!")
    main(
        device,
        config,
        config["num_pred"],
        config["rot"],
        config["num_in"],
        config["n_layers"],
        config["hidden_size"],
        config["epochs"],
        config["lr"],
        config["dropout"],
    )  # pass num_pred=params["num_pred"], etc.


search_space = {
    "num_pred": tune.choice([6, 12, 24]),
    "rot": tune.choice([4]),
    "num_in": tune.choice([50, 110, 220]),
    "n_layers": tune.choice([3, 5, 7]),
    "hidden_size": tune.choice([64, 140, 256]),
    "epochs": tune.choice([500]),
    "lr": tune.loguniform(1e-3, 1e-1),
    "dropout": tune.choice([0.0, 0.1, 0.3]),
}
algo = OptunaSearch()
scheduler = ASHAScheduler(max_t=500, grace_period=10, reduction_factor=3)
ray.init(num_cpus=16,
    num_gpus=4,include_dashboard=False)
tuner = tune.Tuner(  # ③
    tune.with_resources(objective, {"cpu": 4, "gpu": 1}),
    tune_config=tune.TuneConfig(
        metric="score",
        mode="min",
        search_alg=algo,
        scheduler=scheduler,
        num_samples=20,
        trial_dirname_creator=lambda trial: str(trial.trial_id),
    ),
    run_config=tune.RunConfig(
        stop={"training_iteration": 500},
    ),
    param_space=search_space,
)
results = tuner.fit()
print("Best config is:", results.get_best_result().config)
