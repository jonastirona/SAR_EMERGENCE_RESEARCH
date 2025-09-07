import os
import sys
import time
import warnings

import numpy as np
import torch
import wandb
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from functions import (
    PlateauStopper,
    prepare_dataset,
    train_epoch,
    validate_model,
    isVanillaLSTM,
    BASE_PATH,
    RESULTS_PATH,
)
from ray import tune
import ray
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler
from eval import eval_AR_emergence as eval

if isVanillaLSTM:
    from functions import VanillaLSTM as LSTM

    model_type = "VanillaLSTM"
else:
    from functions import LSTM as LSTM

    model_type = "LSTM"


# Assume these are defined in a 'functions.py' file or similar
# from functions import LSTM, lstm_ready, min_max_scaling

warnings.filterwarnings("ignore")
os.makedirs(RESULTS_PATH, exist_ok=True)  # Ensure the results directory exists


def main(config):
    """Main function to run the experiment."""
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Runs on: {device}")

    # Initialize wandb
    wandb.init(
        project=f"{model_type},global_min_max",
        entity=os.environ.get("WANDB_ENTITY"),
        config=config,
        name=f"{model_type}_pred{config['num_pred']}_r{config['rid_of_top']}_i{config['num_in']}_n{config['num_layers']}_h{config['hidden_size']}_e{config['n_epochs']}_l{config['learning_rate']:.5f}_d{config['dropout']:.2f}",
        notes=f"{model_type} training with lr={config['learning_rate']}, dropout={config['dropout']}",
    )

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
    x_train, y_train, _, input_size, m_scale, flux_scale, cont_int_scale = (
        prepare_dataset(
            train_ars,
            9,
            config["rid_of_top"],
            config["num_in"],
            config["num_pred"],
        )
    )

    print("Loading and preparing test data...")
    val_ars = [11462, 11521, 11907, 12219, 12271, 12275, 12567]
    x_val, y_val, last_all, _, _, _, _ = prepare_dataset(
        val_ars,
        9,
        config["rid_of_top"],
        config["num_in"],
        config["num_pred"],
        m_scale,
        flux_scale,
        cont_int_scale,
    )

    if x_train is None or x_val is None:
        print("Could not create datasets. Exiting.")
        return

    train_loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=config["batch_size"], shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(x_val, y_val, last_all), batch_size=config["batch_size"], shuffle=False
    )

    # --- Model & Optimizer ---
    model = LSTM(
        input_size,
        config["hidden_size"],
        config["num_layers"],
        config["num_pred"],
        dropout=config["dropout"],
    ).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.2, patience=10)
    model_name = f"{model_type}pred{config['num_pred']}_r{config['rid_of_top']}_i{config['num_in']}_n{config['num_layers']}_h{config['hidden_size']}_e{config['n_epochs']}_lr{config['learning_rate']:.8f}_d{config['dropout']}.pth"

    # --- Training Loop ---
    print("Starting training...")
    for epoch in range(config["n_epochs"]):
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_rmse = validate_model(model, val_loader, loss_fn, device)

        lr = scheduler.get_last_lr()[0]
        scheduler.step(val_rmse)

        log_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "learning_rate": float(lr),
            "RMSE": val_rmse,
        }
        # if val_rmse <= 0.07:
        #     result = BASE_PATH + f"SAR_EMERGENCE_RESEARCH/lstm/results/{val_rmse:.8f}"
        #     os.makedirs(result, exist_ok=True)  # Ensure the results directory exists
        #     model_path = os.path.join(result, model_name)
        #     torch.save(model.state_dict(), model_path)
        #     print(f"Model saved to {model_path}")

        #     model_artifact = wandb.Artifact(
        #         name=f"RMSE-{val_rmse:.8f}-{model_type}-model-{wandb.run.id}",
        #         type="model",
        #         description=f"{model_type} Model for SAR emergence prediction",
        #         metadata=config,
        #     )
        #     model_artifact.add_file(model_path)
        #     wandb.log_artifact(model_artifact)
        # print(log_metrics)
        wandb.log(log_metrics)
        tune.report(log_metrics)

    # --- Save Model & Artifacts ---
    model_name = f"pred{config['num_pred']}_r{config['rid_of_top']}_i{config['num_in']}_n{config['num_layers']}_h{config['hidden_size']}_e{config['n_epochs']}_lr{config['learning_rate']:.8f}_d{config['dropout']}.pth"
    model_path = os.path.join(RESULTS_PATH, model_name)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    model_artifact = wandb.Artifact(
        name=f"lstm-model-{wandb.run.id}",
        type="model",
        description="LSTM Model for SAR emergence prediction",
        metadata=config,
    )
    model_artifact.add_file(model_path)
    wandb.log_artifact(model_artifact)

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
    # For this refactoring to be fully functional, you must provide
    # the implementations for these functions from your 'functions.py' file.
    config = parse_args()
    search_space = {
        "num_pred": tune.choice([12]),
        "rid_of_top": tune.choice([4]),
        "num_in": tune.choice([110]),
        "num_layers": tune.choice([1]),
        "hidden_size": tune.choice([10, 32, 64, 128, 150]),
        "n_epochs": tune.choice([500]),
        "learning_rate": tune.loguniform(1e-5, 1e-3),
        "dropout": tune.choice([0, 0.01, 0.1]),
        "batch_size": tune.choice([4, 8, 16, 32]),
    }
    algo = OptunaSearch()
    scheduler = ASHAScheduler(max_t=500, grace_period=10, reduction_factor=3)

    custom_stopper = PlateauStopper(
        "RMSE", min_epochs=50, patience=10, min_improvement_percent=0.5
    )

    ray.init(num_cpus=4, num_gpus=2, include_dashboard=False)
    tuner = tune.Tuner(  # ③
        tune.with_resources(main, {"gpu": 1}),
        tune_config=tune.TuneConfig(
            metric="RMSE",
            mode="min",
            search_alg=algo,
            scheduler=scheduler,
            num_samples=config["sample_size"],
            trial_dirname_creator=lambda trial: str(trial.trial_id),
        ),
        run_config=tune.RunConfig(
            stop=custom_stopper,
        ),
        param_space=search_space,
    )
    results = tuner.fit()
    print("Best config is:", results.get_best_result().config)
