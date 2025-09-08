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

    rot = 4
    num_in = 110
    num_pred = 12

    model_name = f"{model_type}_n{config['num_layers']}_h{config['hidden_size']}_lr{config['learning_rate']:.8f}_d{config['dropout']}_t{config['teacher_forcing_ratio']}_a{config['alpha']}"
    # Initialize wandb
    wandb.init(
        project=f"{model_type},LSTM with magnitude and derivative loss ",
        entity=os.environ.get("WANDB_ENTITY"),
        config=config,
        name=f"{model_name}",
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
            rot,
            num_in,
            num_pred,
        )
    )

    print("Loading and preparing test data...")
    val_ars = [11462, 11521, 11907, 12219, 12271, 12275, 12567]
    x_val, y_val, last_all, _, _, _, _ = prepare_dataset(
        val_ars,
        9,
        rot,
        num_in,
        num_pred,
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
        TensorDataset(x_val, y_val),
        batch_size=config["batch_size"],
        shuffle=False,
    )

    # --- Model & Optimizer ---
    model = LSTM(
        input_size,
        config["hidden_size"],
        config["num_layers"],
        num_pred,
        dropout=config["dropout"],
    ).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.2, patience=10)

    # --- Training Loop ---
    print("Starting training...")
    for epoch in range(config["n_epochs"]):
        train_loss = train_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
            config["teacher_forcing_ratio"],
            config["alpha"],
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
    model_path = os.path.join(RESULTS_PATH, model_name + ".pth")
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
    # Define the search space from the section above
    search_space = {
        "learning_rate": tune.loguniform(1e-5, 1e-2),
        "alpha": tune.choice([0.1, 0.3, 0.5, 0.7, 0.9]),
        "teacher_forcing_ratio": tune.choice([0.1, 0.25, 0.5]),
        "hidden_size": tune.choice([32, 64, 128]),
        "num_layers": tune.choice([2, 3, 4]),
        "dropout": tune.choice([0.1, 0.2, 0.3]),
        "batch_size": tune.choice([32, 64]),
        # Add any other fixed parameters your train function needs
        "n_epochs": 100,  # Example fixed parameter
    }

    # Scheduler to early-stop bad trials
    scheduler = ASHAScheduler(
        metric="RMSE",
        mode="min",
        grace_period=10,  # Min epochs before a trial can be stopped
        reduction_factor=2,
    )

    # Search algorithm
    search_alg = OptunaSearch(metric="RMSE", mode="min")

    # Set up the Tuner
    tuner = tune.Tuner(
        main,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            num_samples=50,  # Number of different hyperparameter combinations to try
            scheduler=scheduler,
            search_alg=search_alg,
        ),
        run_config=ray.train.RunConfig(
            name="lstm_hyperparameter_search",
            stop={"training_iteration": 100},  # Max epochs per trial
        ),
    )

    # Run the hyperparameter search
    results = tuner.fit()

    # Get the best result
    best_config = results.get_best_result(metric="RMSE", mode="min").config
    print("Best config found: ", best_config)
