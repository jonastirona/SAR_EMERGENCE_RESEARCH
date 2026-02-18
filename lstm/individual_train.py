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
    train_epochHybridVanillaLSTM,
    train_epochHybridLSTM,
    validate_model,
    RESULTS_PATH,
)

from functions import VanillaLSTM
from functions import LSTM


warnings.filterwarnings("ignore")
os.makedirs(RESULTS_PATH, exist_ok=True)


# --- Main Execution ---
def main(config):
    """Main function to run the experiment."""
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
    # row indexes
    tile_weights = [1.0, 1.0, 1.0, 1.0, 5.0, 1.0, 1.0, 1.0, 1.0]

    (
        x_train,
        y_train,
        _,
        weights_train,
        input_size,
        m_scale,
        flux_scale,
        cont_int_scale,
    ) = prepare_dataset(
        train_ars,
        9,
        config["rid_of_top"],
        config["num_in"],
        config["num_pred"],
        tile_weights=tile_weights,
    )

    print("Loading and preparing test data...")
    val_ars = [11462, 11521, 11907, 12219, 12271, 12275, 12567]
    x_val, y_val, last_val, weights_val, _, _, _ = prepare_dataset(
        val_ars,
        9,
        config["rid_of_top"],
        config["num_in"],
        config["num_pred"],
        m_scale,
        flux_scale,
        cont_int_scale,
        tile_weights=tile_weights,
    )

    if x_train is None or x_val is None:
        print("Could not create datasets. Exiting.")
        return

    train_loader = DataLoader(
        TensorDataset(x_train, y_train, weights_train),
        batch_size=config["batch_size"],
        shuffle=False,
    )
    val_loader = DataLoader(
        TensorDataset(x_val, y_val, weights_val),
        batch_size=config["batch_size"],
        shuffle=False,
    )

    # --- Model & Optimizer ---
    ModelClass = LSTM if config["model_type"] == "LSTM" else VanillaLSTM
    model = ModelClass(
        input_size,
        config["hidden_size"],
        config["num_layers"],
        config["num_pred"],
        dropout=config["dropout"],
    ).to(device)
    print(config)
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
        # Teacher forcing decay: linearly from initial value to 0 over 25 epochs
        grace_period = 25
        initial_tf = config.get("tfr", 0)
        teacher_ratio = max(0.0, initial_tf * (1 - epoch / grace_period))

        if config["model_type"] == "LSTM":
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
                model, train_loader, loss_fn, optimizer, device, config["alpha"]
            )
        val_rmse = validate_model(model, val_loader, device)

        lr = scheduler.get_last_lr()[0]
        scheduler.step(val_rmse)

        log_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "learning_rate": float(lr),
            "RMSE": val_rmse,
        }
        print(log_metrics)

    # --- Save Model & Artifacts ---
    model_name = f"{config['model_type']}{config['num_pred']}_r{config['rid_of_top']}_i{config['num_in']}_n{config['num_layers']}_h{config['hidden_size']}_e{config['n_epochs']}_lr{config['learning_rate']:.8f}_d{config['dropout']}.pth"
    model_path = os.path.join(RESULTS_PATH, model_name)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    end_time = time.time()
    print(f"Elapsed time: {(end_time - start_time) / 60:.2f} minutes")
    # wandb.finish()


def parse_args():
    """Parses command-line arguments."""
    if len(sys.argv) != 9:
        print(
            "Usage: python train_one_epoch.p <num_layers> <hidden_size> <n_epochs> <learning_rate> <dropout> <batch_size> <weight_decay> <model_type>"
        )
        sys.exit(1)

    try:
        config = {
            "num_layers": int(sys.argv[1]),
            "hidden_size": int(sys.argv[2]),
            "n_epochs": int(sys.argv[3]),
            "learning_rate": float(sys.argv[4]),
            "dropout": float(sys.argv[5]),
            "batch_size": int(sys.argv[6]),
            "weight_decay": float(sys.argv[7]),
            "model_type": sys.argv[8],
            "alpha": 0.9,
            "rid_of_top": 0,
            "num_in": 110,
            "num_pred": 12,
            "tfr": 0.5,
        }
        return config
    except (ValueError, IndexError) as e:
        print(f"Error parsing arguments: {e}")
        sys.exit(1)


if __name__ == "__main__":
    config = parse_args()
    main(config)
