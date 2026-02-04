import os
import sys
import time
import warnings

import torch
from math import log

# os.environ["WANDB_MODE"] = "disabled"

import wandb
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from functions import (
    prepare_dataset,
    train_epochHybridLSTM,
    train_epochHybridVanillaLSTM,
    train_epochTeacherForcingLSTM,
    train_epoch,
    validate_model,
    load_all_ar_data,
    RESULTS_PATH,
    MODELS_PATH,
    VanillaLSTM,
    LSTM,
)
from hyperopt import hp
from ray import tune
import ray
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.schedulers import ASHAScheduler
from ray.tune.stopper import TrialPlateauStopper


# Assume these are defined in a 'functions.py' file or similar
# from functions import LSTM, lstm_ready, min_max_scaling

warnings.filterwarnings("ignore")
os.makedirs(RESULTS_PATH, exist_ok=True)  # Ensure the results directory exists

rot = 0
num_in = 110
num_pred = 12
# --- Data Loading ---
print("Loading and preparing training data...")
train_ars = [
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

# PRE-LOAD DATA ONCE
print("Pre-loading all AR data...")
train_data_raw = load_all_ar_data(train_ars, 9, rot)

print("Loading and preparing test data...")
val_ars = [11462, 11521, 11907, 12219, 12271, 12275, 12567]
# Val data can be prepared once as it doesn't depend on hyperparams (unless we change rot/num_in/num_pred which are constants here)
# However, scalers come from training data.
# The original code loaded scaled training data then valid data using those scalers.
# Since scalers only depend on raw values, we can compute them once if we assume the training split is constant.
# BUT, prepare_dataset calculates scalers if not provided.

# Let's keep validation loading inside or pass raw validation data too?
# Validation dataset doesn't change between trials, so we can prepare it fully ONCE.
# Wait, prepare_dataset returns scalers. We need those scalers to prepare validation data.
# So we run prepare_dataset on raw training data ONCE to get scalers and initial X/y (with default weights 1.0 maybe?)
# Actually prepare_dataset logic is: calculate scalers -> scale -> create sequences.

# To be safe and support the flow:
# We will pass raw training data to the loop.
# Validation data: we can pre-load raw validation data too.

val_data_raw = load_all_ar_data(val_ars, 9, rot)  # Load raw val data

if train_data_raw[0] is None:
    print("Could not create datasets. Exiting.")
    sys.exit()


def main(config, train_data_raw, val_data_raw):  # Accept raw data
    model_type = config["model"]["model"]
    lossFn = config["lossFn"]["lossFn"]
    """Main function to run the experiment."""
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Runs on: {device}")

    model_name = f"{model_type}_n{config['num_layers']}_h{config['hidden_size']}_lr{config['learning_rate']:.8f}_d{config['dropout']}_w{config['weight_decay']}_{'shuffle' if config['shuffle'] else 'noshuffle'}_p{config['proportion']}"
    # Initialize wandb
    wandb.init(
        project="Weighted Tiles Test | Only diff proportion parameter",
        entity=os.environ.get("WANDB_ENTITY"),
        config=config,
        name=f"{model_name}",
        notes="",
    )

    best_val_rmse = float("inf")

    # --- Compute Weights Dynamically ---
    proportion = config["proportion"]
    # size = 9 # Implicit in prepare_dataset call if we used it, but here we just pass proportion.
    # Wait, prepare_dataset needs tile_weights list.
    # Logic from original grid_search:
    # weights[rows < 4] = proportion
    # weights[rows >= (size - 4)] = proportion
    # So for size 9, rows 0,1,2,3 are weighted, 5,6,7,8 are weighted. Row 4 is 1.0?
    # Index 0-3: prop. Index 5-8: prop. Index 4: 1.0 (default).

    # Construct tile_weights list of length 9
    tile_weights = [proportion] * 4 + [1.0] + [proportion] * 4

    # Generate Training Dataset
    (
        x_train,
        y_train,
        _,
        weights_train,  # These will be correctly weighted now
        tile_indices_train,
        input_size,
        m_scale,
        flux_scale,
        cont_int_scale,
    ) = prepare_dataset(
        None,  # ar_list is None because we use pre_loaded_data
        9,
        rot,
        num_in,
        num_pred,
        tile_weights=tile_weights,
        pre_loaded_data=train_data_raw,
    )

    # Generate Validation Dataset (using scalers from train)
    x_val, y_val, last_all, weights_val, tile_indices_val, _, _, _, _ = prepare_dataset(
        None,
        9,
        rot,
        num_in,
        num_pred,
        m_scale,
        flux_scale,
        cont_int_scale,
        pre_loaded_data=val_data_raw,
    )

    train_dataset = TensorDataset(
        x_train, y_train, weights_train
    )  # Use returned weights

    # Validation dataset commonly uses equal weights or we just ignore them in validation metric (RMSE)
    # The original code set validation weights to 1.0 (implicitly via prepare_dataset default)

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=config["shuffle"],
    )

    val_dataset = TensorDataset(x_val, y_val, weights_val)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
    )

    # --- Model & Optimizer ---
    model = None
    if model_type == "LSTM":
        model = LSTM(
            input_size,
            config["hidden_size"],
            config["num_layers"],
            num_pred,
            dropout=config["dropout"],
        ).to(device)
    else:
        model = VanillaLSTM(
            input_size,
            config["hidden_size"],
            config["num_layers"],
            num_pred,
            dropout=config["dropout"],
        ).to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.2, patience=10)

    # --- Training Loop ---
    print("Starting training...")
    for epoch in range(config["n_epochs"]):
        train_loss = None
        if model_type == "LSTM":
            if lossFn == "hybrid":
                train_loss = train_epochHybridLSTM(
                    model,
                    train_loader,
                    loss_fn,
                    optimizer,
                    device,
                    config["model"]["teacher_forcing_ratio"],
                    config["lossFn"]["alpha"],
                )
            else:
                train_loss = train_epochTeacherForcingLSTM(
                    model,
                    train_loader,
                    loss_fn,
                    optimizer,
                    device,
                    config["model"]["teacher_forcing_ratio"],
                )
        else:
            if lossFn == "hybrid":
                train_loss = train_epochHybridVanillaLSTM(
                    model,
                    train_loader,
                    loss_fn,
                    optimizer,
                    device,
                    config["lossFn"]["alpha"],
                )
            else:
                train_loss = train_epoch(
                    model,
                    train_loader,
                    loss_fn,
                    optimizer,
                    device,
                )
        val_rmse = validate_model(model, val_loader, device)

        lr = scheduler.get_last_lr()[0]
        scheduler.step(val_rmse)

        log_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            # "validation_loss": val_loss,
            "learning_rate": float(lr),
            "RMSE": val_rmse,
        }

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            # Save strictly for upload purposes
            save_filename = f"{model_name}.pth"
            save_path = os.path.join(MODELS_PATH, save_filename)
            torch.save(model.state_dict(), save_path)

            model_artifact = wandb.Artifact(
                name=f"{model_type}-model-{wandb.run.id}",
                type="model",
                description=f"Best {model_type} model (RMSE: {best_val_rmse:.4f})",
                metadata={**config, "best_rmse": best_val_rmse, "epoch": epoch},
            )
            model_artifact.add_file(save_path)
            wandb.log_artifact(model_artifact)

        wandb.log(log_metrics)
        tune.report(log_metrics)

    # --- Save Model & Artifacts ---
    # We already saved the best model during the loop to wandb.
    # The local file at MODELS_PATH might be the last best one.

    end_time = time.time()
    print(f"Elapsed time: {(end_time - start_time) / 60:.2f} minutes")
    wandb.finish()


def parse_args():
    """Parses command-line arguments."""
    if len(sys.argv) != 2:
        print("Usage: python train_one_epoch.py <grid_search sample_size>")
        sys.exit(1)

    try:
        config = {"sample_size": int(sys.argv[1])}
        return config
    except (ValueError, IndexError) as e:
        print(f"Error parsing arguments: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Define the search space from the section above
    search_space = {
        "learning_rate": hp.loguniform("learning_rate", log(1e-5), log(1e-2)),
        "hidden_size": hp.choice("hidden_size", [2, 4, 8, 16, 32, 64, 128]),
        "num_layers": hp.choice("num_layers", [1, 2, 3, 4]),
        "dropout": hp.choice("dropout", [0, 0.1, 0.2, 0.3]),
        "batch_size": hp.choice("batch_size", [32, 64]),
        "weight_decay": hp.loguniform("weight_decay", log(1e-6), log(1e-3)),
        "n_epochs": 100,
        "proportion": hp.choice("proportion", [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]),
        # Dataset
        "shuffle": hp.choice("shuffle", [True, False]),
        # Model architecture | Conditional Search Space
        "model": hp.choice(
            "model_branch",
            [
                {
                    "model": "VanillaLSTM",
                },
                {
                    "model": "LSTM",
                    "teacher_forcing_ratio": hp.choice(
                        "teacher_forcing_ratio", [0, 0.1, 0.15, 0.25, 0.5]
                    ),
                },
            ],
        ),
        "lossFn": hp.choice(
            "lossFn_branch",
            [
                {
                    "lossFn": "hybrid",
                    "alpha": hp.choice("alpha", [0.1, 0.3, 0.5, 0.7, 0.9]),
                },
                {"lossFn": "value"},
            ],
        ),
    }

    # Scheduler to early-stop bad trials
    scheduler = ASHAScheduler(
        metric="RMSE",
        mode="min",
        grace_period=15,  # Min epochs before a trial can be stopped
        reduction_factor=2,
    )

    # Search algorithm
    search_alg = HyperOptSearch(space=search_space, metric="RMSE", mode="min")

    early_stopper = TrialPlateauStopper(
        metric="RMSE",
        mode="min",
        grace_period=10,  # Number of epochs to wait for improvement
    )

    # Set up the Tuner
    ray.init(num_cpus=32, num_gpus=1, include_dashboard=False, _temp_dir="/tmp/ray")

    # Put large data in Ray object store
    train_data_ref = ray.put(train_data_raw)
    val_data_ref = ray.put(val_data_raw)

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(
                main,
                train_data_raw=train_data_ref,
                val_data_raw=val_data_ref,
            ),
            {"gpu": 32, "cpu": 1},
        ),
        tune_config=tune.TuneConfig(
            num_samples=parse_args()[
                "sample_size"
            ],  # Number of different hyperparameter combinations to try
            scheduler=scheduler,
            search_alg=search_alg,
        ),
        run_config=ray.train.RunConfig(
            name="lstm_hyperparameter_search",
            stop=early_stopper,  # Max epochs per trial
        ),
    )

    # Run the hyperparameter search
    results = tuner.fit()

    # Get the best result
    best_config = results.get_best_result(metric="RMSE", mode="min").config
    print("Best config found: ", best_config)
