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
    train_epoch_emergence_aware,
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


# Assume these are defined in a 'functions.py' file or similar
# from functions import LSTM, lstm_ready, min_max_scaling

warnings.filterwarnings("ignore")
os.makedirs(RESULTS_PATH, exist_ok=True)  # Ensure the results directory exists

rot = 0
num_in = 110
num_pred = 12
# --- Data Loading ---


# Filter ARs to identify which ones exist, to ensure index alignment
def filter_valid_ars(ar_list):
    valid_ars = []
    for ar in ar_list:
        # Check if file exists roughly based on load_ar_data logic
        # We assume if one file exists, others likely do, or we rely on load_ar_data returning None
        # But we need to know BEFORE calling load_all_ar_data to align indices.
        # Check one file:
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

# PRE-LOAD DATA ONCE
print("Pre-loading all AR data...")
train_data_raw = load_all_ar_data(train_ars, 9, rot)

print("Loading and preparing test data...")
val_ars = [11462, 11521, 11907, 12219, 12271, 12275, 12567]
val_ars = filter_valid_ars(val_ars)

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

    model_name = f"{model_type}_n{config['num_layers']}_h{config['hidden_size']}_lr{config['learning_rate']:.8f}_d{config['dropout']}_w{config['weight_decay']}_{'shuffle' if config['shuffle'] else 'noshuffle'}_custom_weights"
    # Initialize wandb
    wandb.init(
        project="Active Region RMSE | emergence timing missed prediction fix",
        entity=os.environ.get("WANDB_ENTITY"),
        config=config,
        name=f"Fixed_W_0.05_1.0_{model_name}",
        notes="Using labeled_regions.json: 1.0 for labeled, 0.05 for others. Granular metrics.",
    )

    best_val_rmse = float("inf")

    # --- Load Labeled Regions ---
    import json

    # Use path relative to the script or project root lookup
    # labeled_regions.json is in parent of lstm dir (project root)
    # DATA_PATH is .../SAR_EMERGENCE_RESEARCH/data
    # So it should be at .../SAR_EMERGENCE_RESEARCH/labeled_regions.json
    json_path = os.path.join(os.path.dirname(DATA_PATH), "labeled_regions.json")
    if not os.path.exists(json_path):
        # Try current directory or parent
        if os.path.exists("labeled_regions.json"):
            json_path = "labeled_regions.json"
        elif os.path.exists("../labeled_regions.json"):
            json_path = "../labeled_regions.json"

    print(f"Loading labeled regions from: {json_path}")
    with open(json_path, "r") as f:
        labeled_regions = json.load(f)

    # We pass the dictionary directly.
    tile_weights = labeled_regions

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
        train_ars,  # ar_list passed explicitly for AR-based weighting
        9,
        rot,
        num_in,
        num_pred,
        tile_weights=tile_weights,
        pre_loaded_data=train_data_raw,
    )

    # Generate Validation Dataset (using scalers from train)
    # Pass tile_weights so validation can compute Active vs Background RMSE
    x_val, y_val, last_all, weights_val, tile_indices_val, _, _, _, _ = prepare_dataset(
        val_ars,
        9,
        rot,
        num_in,
        num_pred,
        m_scale,
        flux_scale,
        cont_int_scale,
        tile_weights=tile_weights,  # Pass weights for metric splitting
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
    n_epochs = config["n_epochs"]
    for epoch in range(n_epochs):
        train_loss = None

        if lossFn == "emergence":
            # Curriculum annealing: k ramps from 10 to 100 over training
            k = 10.0 + (epoch / max(n_epochs - 1, 1)) * 90.0
            teacher_forcing = (
                config["model"]["teacher_forcing_ratio"]
                if model_type == "LSTM"
                else None
            )
            train_loss = train_epoch_emergence_aware(
                model,
                train_loader,
                loss_fn,
                optimizer,
                device,
                alpha=config["lossFn"]["alpha"],
                gamma=config["lossFn"]["gamma"],
                k=k,
                teacher_forcing=teacher_forcing,
            )
        elif model_type == "LSTM":
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
        val_metrics = validate_model(model, val_loader, device)
        val_rmse = val_metrics["RMSE"]
        val_deriv_rmse = val_metrics["Deriv_RMSE"]
        val_weighted_rmse = val_metrics["Weighted_RMSE"]
        val_weighted_deriv_rmse = val_metrics["Weighted_Deriv_RMSE"]
        val_active_rmse = val_metrics["Active_RMSE"]
        val_active_deriv_rmse = val_metrics["Active_Deriv_RMSE"]
        val_bg_rmse = val_metrics["Background_RMSE"]
        val_emergence_mae = val_metrics["Emergence_Timing_MAE"]
        val_detection_rate = val_metrics["Emergence_Detection_Rate"]
        val_false_alarm_rate = val_metrics["Emergence_False_Alarm_Rate"]

        # Composite metric: 70% derivative accuracy, 20% emergence timing, 10% detection reliability
        # Detection penalty ensures models that never predict emergence are punished
        detection_penalty = 0.2 * (1.0 - val_detection_rate)
        composite_score = (
            0.7 * val_active_deriv_rmse
            + 0.2 * (val_emergence_mae * 0.05)
            + detection_penalty
        )

        lr = scheduler.get_last_lr()[0]
        # Schedule on composite score
        scheduler.step(composite_score)

        log_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "learning_rate": float(lr),
            "RMSE": val_rmse,
            "Deriv_RMSE": val_deriv_rmse,
            "Weighted_RMSE": val_weighted_rmse,
            "Weighted_Deriv_RMSE": val_weighted_deriv_rmse,
            "Active_RMSE": val_active_rmse,
            "Active_Deriv_RMSE": val_active_deriv_rmse,
            "Background_RMSE": val_bg_rmse,
            "Emergence_Timing_MAE": val_emergence_mae,
            "Emergence_Detection_Rate": val_detection_rate,
            "Emergence_False_Alarm_Rate": val_false_alarm_rate,
            "Composite_Score": composite_score,
        }

        # Save best model based on composite score
        if composite_score < best_val_rmse:
            best_val_rmse = composite_score
            # Save strictly for upload purposes
            save_filename = f"{model_name}.pth"
            save_path = os.path.join(MODELS_PATH, save_filename)
            torch.save(model.state_dict(), save_path)

            model_artifact = wandb.Artifact(
                name=f"{model_type}-model-{wandb.run.id}",
                type="model",
                description=f"Best {model_type} model (Composite: {best_val_rmse:.4f})",
                metadata={
                    **config,
                    "best_rmse": val_rmse,
                    "best_deriv_rmse": val_deriv_rmse,
                    "best_weighted_rmse": val_weighted_rmse,
                    "best_weighted_deriv_rmse": val_weighted_deriv_rmse,
                    "best_active_deriv_rmse": val_active_deriv_rmse,
                    "best_emergence_timing_mae": val_emergence_mae,
                    "best_detection_rate": val_detection_rate,
                    "best_false_alarm_rate": val_false_alarm_rate,
                    "best_composite_score": best_val_rmse,
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
        "lossFn": hp.choice(
            "lossFn_branch",
            [
                {
                    "lossFn": "hybrid",
                    "alpha": hp.choice("alpha", [0.1, 0.3, 0.5, 0.7, 0.9]),
                },
                {"lossFn": "value"},
                {
                    "lossFn": "emergence",
                    "alpha": hp.choice("alpha_emergence", [0.3, 0.5, 0.7]),
                    "gamma": hp.choice("gamma", [0.05, 0.1, 0.3, 0.5]),
                },
            ],
        ),
    }

    # Scheduler to early-stop bad trials
    scheduler = ASHAScheduler(
        metric="Composite_Score",
        mode="min",
        grace_period=30,  # Min epochs before ASHA can kill a trial
        reduction_factor=2,
    )

    # Search algorithm
    search_alg = HyperOptSearch(
        space=search_space, metric="Composite_Score", mode="min"
    )

    early_stopper = TrialPlateauStopper(
        metric="Composite_Score",
        mode="min",
        num_results=8,  # Check last 8 values for plateau (default was 4)
        grace_period=25,  # Don't check until at least 25 epochs
        std=0.001,  # Require truly flat metric (default 0.01 was too generous)
    )

    # Set up the Tuner
    ray.init(num_cpus=32, num_gpus=2, include_dashboard=False, _temp_dir="/tmp/ray")

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
            {"gpu": 1 / 8, "cpu": 1},
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
    best_config = results.get_best_result(metric="Composite_Score", mode="min").config
    print("Best config found: ", best_config)
