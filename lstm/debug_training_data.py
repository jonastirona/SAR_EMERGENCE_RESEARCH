import os
import sys
import torch
import numpy as np
import json
from functions import prepare_dataset, load_all_ar_data, DATA_PATH

# Configuration
print(f"DEBUG: Using DATA_PATH: {DATA_PATH}")
rot = 0
num_in = 110
num_pred = 12
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


def filter_valid_ars(ar_list):
    valid_ars = []
    for ar in ar_list:
        pm_path = os.path.join(DATA_PATH, f"AR{ar}", f"mean_pmdop{ar}_flat.npz")
        if os.path.exists(pm_path):
            valid_ars.append(ar)
    return valid_ars


# 1. Filter ARs
print("Filtering ARs...")
train_ars = filter_valid_ars(train_ars)
print(f"Valid ARs: {len(train_ars)}")

# 2. Load Labeled Regions
print("Loading labeled_regions.json...")
# Try multiple paths
paths = [
    "labeled_regions.json",
    "../labeled_regions.json",
    os.path.join(os.path.dirname(DATA_PATH), "labeled_regions.json"),
]
labeled_regions = None
for p in paths:
    if os.path.exists(p):
        print(f"Found labeled_regions.json at {p}")
        with open(p, "r") as f:
            labeled_regions = json.load(f)
        break

if labeled_regions is None:
    print("ERROR: Could not find labeled_regions.json")
    sys.exit(1)

# 3. Load Raw Data
print("Loading raw data...")
train_data_raw = load_all_ar_data(train_ars, 9, rot)

# 4. Prepare Dataset
print("Preparing dataset with weights...")
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
    tile_weights=labeled_regions,
    pre_loaded_data=train_data_raw,
)

# 5. Analyze
print("\n--- Analysis ---")
print(f"X shape: {x_train.shape}")
print(f"Y shape: {y_train.shape}")
print(f"Weights shape: {weights_train.shape}")

y_np = y_train.numpy()
w_np = weights_train.numpy()

print(f"\nTarget Stats (Y):")
print(f"  Mean: {np.mean(y_np):.6f}")
print(f"  Std:  {np.std(y_np):.6f}")
print(f"  Min:  {np.min(y_np):.6f}")
print(f"  Max:  {np.max(y_np):.6f}")
print(
    f"  Zeros: {np.sum(y_np == 0)} / {y_np.size} ({np.sum(y_np == 0) / y_np.size * 100:.2f}%)"
)

print(f"\nWeight Stats:")
unique, counts = np.unique(w_np, return_counts=True)
print(f"  Values: {dict(zip(unique, counts))}")

weighted_mean_mse = np.mean((y_np - np.mean(y_np)) ** 2)  # Variance
print(f"  Variance of Y (MSE if predicting mean): {weighted_mean_mse:.6f}")

# Check if weights are actually matching
print("\nchecking first 20 samples...")
for i in range(20):
    print(f"  Idx {i}: Tile {int(tile_indices_train[i].item())} -> Weight {w_np[i]}")
