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
    validate_model,
    load_all_ar_data,
    RESULTS_PATH,
    MODELS_PATH,
    DATA_PATH,
    VanillaLSTM,
    LSTM,
)
from hyperopt import hp
from ray import tune
import ray
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.schedulers import ASHAScheduler
from ray.tune.stopper import TrialPlateauStopper


warnings.filterwarnings("ignore")
os.makedirs(RESULTS_PATH, exist_ok=True)

rot = 0
num_in = 110
num_pred = 12
# --- Data Loading ---


# Filter ARs to ensure data exists before loading
def filter_valid_ars(ar_list):
    valid_ars = []
    for ar in ar_list:
        pm_path = os.path.join(DATA_PATH, f"AR{ar}", f"mean_pmdop{ar}_flat.npz")
        if os.path.exists(pm_path):
            valid_ars.append(ar)
        else:
            print(f"Warning: Data for AR {ar} not found at {pm_path}. Skipping.")
    return valid_ars


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

# Pre-load data to avoid repeated reads during search
print("Pre-loading training data...")
train_data_raw = load_all_ar_data(train_ars, 9, rot)

print("Preparing validation data...")
val_ars = filter_valid_ars([11462, 11521, 11907, 12219, 12271, 12275, 12567])
val_data_raw = load_all_ar_data(val_ars, 9, rot)

if train_data_raw[0] is None:
    print("Could not create datasets. Exiting.")
    sys.exit()


def main(config, train_data_raw, val_data_raw):  # Accept raw data
    model_type = config["model"]["model"]
    """Main function to run the experiment."""
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Runs on: {device}")

    model_name = f"{model_type}_n{config['num_layers']}_h{config['hidden_size']}_lr{config['learning_rate']:.8f}_d{config['dropout']}_w{config['weight_decay']}_a{config['alpha']}_{'shuffle' if config['shuffle'] else 'noshuffle'}"
    # Initialize wandb
    wandb.init(
        project="Active Region RMSE | 1 ",
        entity=os.environ.get("WANDB_ENTITY"),
        config=config,
        name=f"Fixed_W_0.05_1.0_{model_name}",
        notes="Using labeled_regions.json: 1.0 for labeled, 0.05 for others. Granular metrics.",
    )

    best_val_rmse = float("inf")

    # Load labeled regions for weighting
    import json

    json_path = os.path.join(os.path.dirname(DATA_PATH), "labeled_regions.json")
    if not os.path.exists(json_path):
        for candidate in ["labeled_regions.json", "../labeled_regions.json"]:
            if os.path.exists(candidate):
                json_path = candidate
                break

    print(f"Loading weights from: {json_path}")
    with open(json_path, "r") as f:
        tile_weights = json.load(f)

    # Scale and prepare sequences
    (
        x_train,
        y_train,
        _,
        weights_train,
        tile_indices_train,
        input_size,
        m_scale,
        flux_scale,
        cont_int_scale,
    ) = prepare_dataset(
        train_ars,
        9,
        rot,
        num_in,
        num_pred,
        tile_weights=tile_weights,
        pre_loaded_data=train_data_raw,
    )

    x_val, y_val, _, weights_val, _, _, _, _, _ = prepare_dataset(
        val_ars,
        9,
        rot,
        num_in,
        num_pred,
        m_scale,
        flux_scale,
        cont_int_scale,
        tile_weights=tile_weights,
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
        # Teacher forcing decay: linearly from initial value to 0 over grace period
        grace_period = 25
        initial_tf = config["model"].get("teacher_forcing_ratio", 0)
        teacher_ratio = max(0.0, initial_tf * (1 - epoch / grace_period))

        if model_type == "LSTM":
            train_loss = train_epochHybridLSTM(
                model,
                train_loader,
                loss_fn,
                optimizer,
                device,
                teacher_ratio,
                config["alpha"],
            )
        else:
            train_loss = train_epochHybridVanillaLSTM(
                model,
                train_loader,
                loss_fn,
                optimizer,
                device,
                config["alpha"],
            )
        val_metrics = validate_model(model, val_loader, device)
        val_rmse = val_metrics["RMSE"]
        val_grad_rmse = val_metrics["Grad_RMSE"]
        val_weighted_rmse = val_metrics["Weighted_RMSE"]
        val_weighted_grad_rmse = val_metrics["Weighted_Grad_RMSE"]
        val_active_rmse = val_metrics["Active_RMSE"]
        val_active_grad_rmse = val_metrics["Active_Grad_RMSE"]
        val_bg_rmse = val_metrics["Background_RMSE"]

        lr = scheduler.get_last_lr()[0]
        # Optimize primarily for Active Grad RMSE
        scheduler.step(val_active_grad_rmse)

        log_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "learning_rate": float(lr),
            "RMSE": val_rmse,
            "Grad_RMSE": val_grad_rmse,
            "Weighted_RMSE": val_weighted_rmse,
            "Weighted_Grad_RMSE": val_weighted_grad_rmse,
            "Active_RMSE": val_active_rmse,
            "Active_Grad_RMSE": val_active_grad_rmse,
            "Background_RMSE": val_bg_rmse,
        }

        # Save best model
        if val_active_grad_rmse < best_val_rmse:
            best_val_rmse = val_active_grad_rmse
            save_path = os.path.join(MODELS_PATH, f"{model_name}.pth")
            torch.save(model.state_dict(), save_path)

            model_artifact = wandb.Artifact(
                name=f"{model_type}-model-{wandb.run.id}-epoch{epoch}",
                type="model",
                description=f"Best {model_type} model (Active Grad RMSE: {best_val_rmse:.4f})",
                metadata={
                    **config,
                    "best_rmse": val_rmse,
                    "best_grad_rmse": val_grad_rmse,
                    "best_weighted_rmse": val_weighted_rmse,
                    "best_weighted_grad_rmse": val_weighted_grad_rmse,
                    "best_active_grad_rmse": best_val_rmse,
                    "epoch": epoch,
                },
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
        # "proportion": hp.choice("proportion", [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]),
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
        # alpha=1.0 -> value only, alpha=0.0 -> gradient only, between -> hybrid
        "alpha": hp.choice("alpha", [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]),
    }

    # Scheduler to early-stop bad trials
    scheduler = ASHAScheduler(
        metric="Active_Grad_RMSE",  # Optimize for active regions!
        mode="min",
        grace_period=30,  # Min epochs before ASHA can kill a trial
        reduction_factor=2,
    )

    # Search algorithm
    search_alg = HyperOptSearch(
        space=search_space, metric="Active_Grad_RMSE", mode="min"
    )

    early_stopper = TrialPlateauStopper(
        metric="Active_Grad_RMSE",
        mode="min",
        num_results=8,  # Check last 8 values for plateau (default was 4)
        grace_period=25,  # Don't check until at least 25 epochs
        std=0.001,  # Require truly flat metric (default 0.01 was too generous)
    )

    # Set up the Tuner
    ray.init(num_cpus=8, num_gpus=1, include_dashboard=False, _temp_dir="/tmp/ray")

    # Pass raw data via Ray object store
    train_data_ref = ray.put(train_data_raw)
    val_data_ref = ray.put(val_data_raw)

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(
                main, train_data_raw=train_data_ref, val_data_raw=val_data_ref
            ),
            {"gpu": 1 / 8, "cpu": 1},
        ),
        tune_config=tune.TuneConfig(
            num_samples=parse_args()["sample_size"],
            scheduler=scheduler,
            search_alg=search_alg,
        ),
        run_config=ray.train.RunConfig(
            name="lstm_hyperparameter_search",
            stop=early_stopper,
        ),
    )

    # Run the hyperparameter search
    results = tuner.fit()

    # Get the best result
    best_config = results.get_best_result(metric="Active_Grad_RMSE", mode="min").config
    print("Best config found: ", best_config)
