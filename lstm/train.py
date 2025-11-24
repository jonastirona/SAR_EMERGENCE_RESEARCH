import os
import sys
import time
import warnings
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from functions import (
    prepare_dataset,
    train_epoch,
    train_epochHybridVanillaLSTM,
    validate_model,
    validate_modelv1,
    validate_modelv2,
    isVanillaLSTM,
    RESULTS_PATH,
    WeightedMSELoss,
)

if isVanillaLSTM:
    from functions import VanillaLSTM as LSTM
else:
    from functions import LSTM as LSTM

# Assume these are defined in a 'functions.py' file or similar
# from functions import LSTM, lstm_ready, min_max_scaling

warnings.filterwarnings("ignore")
os.makedirs(RESULTS_PATH, exist_ok=True)  # Ensure the results directory exists


def main(config):
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Runs on: {device}")

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
            train_ars, 9, config["rid_of_top"], config["num_in"], config["num_pred"]
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

    loss_fn = WeightedMSELoss(high_val_weight=50.0, threshold=0.1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    # ---------------------------------------------------------
    # CRITICAL CHANGE: mode="max"
    # We want to MAXIMIZE lead time.
    # ---------------------------------------------------------
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=10)

    # --- Training Loop ---
    print("Starting training...")
    for epoch in range(config["n_epochs"]):
        train_loss = train_epochHybridVanillaLSTM(
            model, train_loader, loss_fn, optimizer, device, config["alpha"]
        )

        # Unpack both return values
        val_rmse, avg_lead_time = validate_modelv2(model, val_loader, device)

        lr = scheduler.get_last_lr()[0]

        # Step the scheduler based on Lead Time (Max)
        scheduler.step(avg_lead_time)

        log_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "learning_rate": float(lr),
            "RMSE": val_rmse,  # We still watch this
            "Lead_Time_Hrs": avg_lead_time,  # This is what we optimize for
        }
        print(log_metrics)
        # wandb.log(log_metrics)

    # --- Save Model ---
    model_name = f"pred{config['num_pred']}_r{config['rid_of_top']}_i{config['num_in']}_n{config['num_layers']}_h{config['hidden_size']}_e{config['n_epochs']}_lr{config['learning_rate']:.8f}_d{config['dropout']}.pth"
    model_path = os.path.join(RESULTS_PATH, model_name)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    end_time = time.time()
    print(f"Elapsed time: {(end_time - start_time) / 60:.2f} minutes")


def parse_args():
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
            "alpha": 0.7,
        }
        return config
    except (ValueError, IndexError) as e:
        print(f"Error parsing arguments: {e}")
        sys.exit(1)


if __name__ == "__main__":
    config = parse_args()
    main(config)
