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
    train_epochHybrid,
    validate_model,
    isVanillaLSTM,
    BASE_PATH,
    RESULTS_PATH,
)
from ray import tune
import ray
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler


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


# --- Main Execution ---
def main(config):
    """Main function to run the experiment."""
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Runs on: {device}")

    # Initialize wandb
    # wandb.init(
    #     project="LSTM,Future_11,NUM_IN_110,pred_12",
    #     entity=os.environ.get("WANDB_ENTITY"),
    #     config=config,

    #     name=f"LSTM_pred{config['num_pred']}_r{config['rid_of_top']}_i{config['num_in']}_n{config['num_layers']}_h{config['hidden_size']}_e{config['n_epochs']}_l{config['learning_rate']:.5f}_d{config['dropout']:.2f}",
    #     notes=f"LSTM training with lr={config['learning_rate']}, dropout={config['dropout']}",
    # )

    # --- Data Loading ---
    print("Batch size:", config["batch_size"])
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
    x_val, y_val, last_val, _, _, _, _ = prepare_dataset(
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
        TensorDataset(x_val, y_val), batch_size=config["batch_size"], shuffle=False
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

    # --- Training Loop ---
    print("Starting training...")
    for epoch in range(config["n_epochs"]):
        teacher = max(0.0, 0.6 * (1 - epoch / config["n_epochs"]))
        train_loss = train_epochHybrid(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
            teacher_ratio=teacher,
            alpha=0.7,
        )
        val_rmse = validate_model(model, val_loader, device)

        lr = scheduler.get_last_lr()[0]
        scheduler.step(val_rmse)

        # Evaluate every 10 epochs and on the last epoch
        # scores = []
        # for AR in [11698, 11726, 13165, 13179, 13183]:
        #     score = eval(device, AR, False, BASE_PATH, model.state_dict(), **config)
        #     scores.append(score)
        # val_rmse = float(np.mean(scores))

        log_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            # "validation_loss": val_loss,
            "learning_rate": float(lr),
            "RMSE": val_rmse,
        }
        print(log_metrics)
        # wandb.log(log_metrics)

    # --- Save Model & Artifacts ---
    model_name = f"pred{config['num_pred']}_r{config['rid_of_top']}_i{config['num_in']}_n{config['num_layers']}_h{config['hidden_size']}_e{config['n_epochs']}_lr{config['learning_rate']:.8f}_d{config['dropout']}.pth"
    model_path = os.path.join(RESULTS_PATH, model_name)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    scales = {
        "m_scale": m_scale,
        "flux_scale": flux_scale,
        "cont_int_scale": cont_int_scale,
    }
    np.savez(os.path.join(RESULTS_PATH, "model_scales.npz"), **scales)

    # model_artifact = wandb.Artifact(
    #     name=f"lstm-model-{wandb.run.id}",
    #     type="model",
    #     description="LSTM Model for SAR emergence prediction",
    #     metadata=config,
    # )
    # model_artifact.add_file(model_path)
    # wandb.log_artifact(model_artifact)

    end_time = time.time()
    print(f"Elapsed time: {(end_time - start_time) / 60:.2f} minutes")
    # wandb.finish()


def parse_args():
    """Parses command-line arguments."""
    if len(sys.argv) != 10:
        print(
            "Usage: python train_one_epoch.py <num_pred> <rid_of_top> <num_in> <num_layers> <hidden_size> <n_epochs> <learning_rate> <dropout> <batch_size>"
        )
        sys.exit(1)

    try:
        config = {
            "num_pred": int(sys.argv[1]),
            "rid_of_top": int(sys.argv[2]),
            "num_in": int(sys.argv[3]),
            "num_layers": int(sys.argv[4]),
            "hidden_size": int(sys.argv[5]),
            "n_epochs": int(sys.argv[6]),
            "learning_rate": float(sys.argv[7]),
            "dropout": float(sys.argv[8]),
            "batch_size": int(sys.argv[9]),
        }
        return config
    except (ValueError, IndexError) as e:
        print(f"Error parsing arguments: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # For this refactoring to be fully functional, you must provide
    # the implementations for these functions from your 'functions.py' file.
    config = parse_args()
    main(config)
