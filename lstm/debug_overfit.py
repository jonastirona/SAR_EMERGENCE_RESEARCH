"""
Overfit verification script.
Tests whether the LSTM can learn AT ALL by training on 1-2 active ARs
with only labeled tiles (weight 1.0, background 0.05).

SUCCESS: Training loss goes to near-zero -> model CAN learn, issue is data balance.
FAILURE: Training loss stays flat -> model/data pipeline is broken.
"""

import os
import json
import torch
import numpy as np
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from functions import (
    prepare_dataset,
    load_all_ar_data,
    train_epoch,
    validate_model,
    DATA_PATH,
    VanillaLSTM,
)

# --- Config ---
OVERFIT_ARS = [11162]  # AR with many labeled tiles (26 tiles)
ROT = 0
NUM_IN = 110
NUM_PRED = 12
SIZE = 9
N_EPOCHS = 50
BATCH_SIZE = 32
HIDDEN_SIZE = 64
NUM_LAYERS = 2
LR = 1e-3
DROPOUT = 0.0  # No dropout for overfitting test

# --- Load data ---
print(f"Loading data for ARs: {OVERFIT_ARS}")
print(f"DATA_PATH: {DATA_PATH}")

# Check files exist
for ar in OVERFIT_ARS:
    pm_path = os.path.join(DATA_PATH, f"AR{ar}", f"mean_pmdop{ar}_flat.npz")
    print(f"  AR {ar}: {pm_path} -> exists={os.path.exists(pm_path)}")

raw_data = load_all_ar_data(OVERFIT_ARS, SIZE, ROT)

# Load labeled regions
json_path = os.path.join(os.path.dirname(DATA_PATH), "labeled_regions.json")
if not os.path.exists(json_path):
    json_path = (
        "labeled_regions.json"
        if os.path.exists("labeled_regions.json")
        else "../labeled_regions.json"
    )
print(f"Loading labeled_regions.json from: {json_path}")
with open(json_path, "r") as f:
    labeled_regions = json.load(f)

# Prepare dataset WITH weights
print("\nPreparing dataset with tile weights...")
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
    OVERFIT_ARS,
    SIZE,
    ROT,
    NUM_IN,
    NUM_PRED,
    tile_weights=labeled_regions,
    pre_loaded_data=raw_data,
)

print(f"\nDataset stats:")
print(f"  X shape: {x_train.shape}")
print(f"  Y shape: {y_train.shape}")
print(f"  Weights shape: {weights_train.shape}")

w_np = weights_train.numpy()
unique, counts = np.unique(w_np, return_counts=True)
print(f"  Weight distribution: {dict(zip(unique, counts))}")

y_np = y_train.numpy()
print(f"  Y mean: {np.mean(y_np):.6f}, std: {np.std(y_np):.6f}")
print(f"  Y variance (baseline MSE): {np.var(y_np):.6f}")

# Subset to ONLY active tiles for overfit test
active_mask = weights_train == 1.0
if active_mask.sum() > 0:
    x_active = x_train[active_mask]
    y_active = y_train[active_mask]
    w_active = weights_train[active_mask]

    print(f"\n  Active-only subset: {x_active.shape[0]} samples")
    print(
        f"  Active Y mean: {np.mean(y_active.numpy()):.6f}, std: {np.std(y_active.numpy()):.6f}"
    )
    print(f"  Active Y variance (baseline MSE): {np.var(y_active.numpy()):.6f}")
else:
    print("\n  WARNING: No active tiles found! Using all data.")
    x_active = x_train
    y_active = y_train
    w_active = weights_train

# --- Create model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")

model = VanillaLSTM(input_size, HIDDEN_SIZE, NUM_LAYERS, NUM_PRED, dropout=DROPOUT).to(
    device
)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=5)

# --- Train on active tiles only ---
dataset = TensorDataset(x_active, y_active, w_active)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f"\n{'=' * 60}")
print(f"OVERFIT TEST: {len(dataset)} samples, {N_EPOCHS} epochs")
print(f"{'=' * 60}")

for epoch in range(N_EPOCHS):
    train_loss = train_epoch(model, loader, loss_fn, optimizer, device)
    lr = scheduler.get_last_lr()[0]
    scheduler.step(train_loss)

    if epoch % 5 == 0 or epoch == N_EPOCHS - 1:
        print(f"  Epoch {epoch:3d}: loss={train_loss:.8f}  lr={lr:.6f}")

print(f"\n{'=' * 60}")
if train_loss < 0.001:
    print("SUCCESS: Model CAN learn. Issue is data balance / weighting.")
elif train_loss < np.var(y_active.numpy()) * 0.5:
    print(
        "PARTIAL SUCCESS: Loss decreased but not near-zero. Model may need more epochs or capacity."
    )
else:
    print("FAILURE: Model cannot learn. Check data pipeline / model architecture.")
print(f"Final loss: {train_loss:.8f}")
print(f"Baseline (variance): {np.var(y_active.numpy()):.8f}")
print(f"{'=' * 60}")
