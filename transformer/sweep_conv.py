import numpy as np
import torch
import torch.nn as nn
import wandb
from typing import Dict, List, Tuple, Any
import logging
import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score
import math
from torch.optim.lr_scheduler import LambdaLR
import signal
import time
import gc

# --- SETUP AND IMPORTS ---
# Add project root to Python path
# Ensure this path logic works with your HPC directory structure
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.append(project_root)
    from transformer.models.st_transformer_conv import SpatioTemporalTransformerConv
    from transformer.functions import lstm_ready as original_lstm_ready, split_sequences
    from transformer.eval import evaluate_models_for_ar
except ImportError as e:
    print(f"Warning: Could not import local modules: {e}. Ensure script is in the correct directory.")
    # Define dummy functions if imports fail, allowing the script to run standalone for testing
    def SpatioTemporalTransformerConv(*args, **kwargs): return nn.Linear(10, 2)
    def original_lstm_ready(*args, **kwargs): return torch.randn(1, 10), torch.randn(1, 1)
    def split_sequences(X, y, n_in, n_out): return np.random.rand(5, n_in, X.shape[1]), np.random.rand(5, n_out)
    def evaluate_models_for_ar(*args, **kwargs): print("Skipping AR evaluation due to import error.")

# Set up logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# Global variable to store normalization statistics
GLOBAL_NORM_STATS = None

# Timeout handler
def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

def set_timeout(seconds):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    return LambdaLR(optimizer, lr_lambda, last_epoch)

def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return r2_score(y_true, y_pred)

def calculate_derivative(time_series: np.ndarray, time_step: float = 1.0) -> np.ndarray:
    return np.gradient(time_series, time_step)

def find_negative_derivative_periods(derivative: np.ndarray, min_duration: int = 4) -> List[Tuple[int, int]]:
    """Find periods where derivative is negative for at least min_duration consecutive time steps."""
    negative_periods = []
    start_idx = None
    
    for i, val in enumerate(derivative):
        if val < 0 and start_idx is None:
            start_idx = i
        elif val >= 0 and start_idx is not None:
            if i - start_idx >= min_duration:
                negative_periods.append((start_idx, i - 1))
            start_idx = None
    
    # Handle case where negative period extends to end
    if start_idx is not None and len(derivative) - start_idx >= min_duration:
        negative_periods.append((start_idx, len(derivative) - 1))
    
    return negative_periods

def detect_emergence_window(observed: np.ndarray, time_step: float = 1.0, negative_duration: int = 4, window_size: int = 24) -> Tuple[int, int]:
    """
    Detect emergence window based on negative derivative periods.
    
    Args:
        observed: Observed time series
        time_step: Time step in hours
        negative_duration: Minimum duration of negative derivative
        window_size: Size of emergence window to return
        
    Returns:
        Tuple of (start_idx, end_idx) for emergence window
    """
    derivative = calculate_derivative(observed, time_step)
    negative_periods = find_negative_derivative_periods(derivative, negative_duration)
    
    if not negative_periods:
        return None, None
    
    # Use the longest negative period
    longest_period = max(negative_periods, key=lambda x: x[1] - x[0])
    start_idx, end_idx = longest_period
    
    # Center the window on the negative period
    center = (start_idx + end_idx) // 2
    window_start = max(0, center - window_size // 2)
    window_end = min(len(observed), window_start + window_size)
    
    return window_start, window_end

def calculate_emergence_metrics(observed: np.ndarray, predicted: np.ndarray, time_step: float = 1.0) -> Dict[str, float]:
    """Calculate emergence detection metrics."""
    emergence_start, emergence_end = detect_emergence_window(observed, time_step)
    
    if emergence_start is None:
        return {
            'emergence_rmse': np.nan,
            'overall_rmse': np.sqrt(np.mean((observed - predicted) ** 2))
        }
    
    # Calculate RMSE for emergence period
    emergence_observed = observed[emergence_start:emergence_end]
    emergence_predicted = predicted[emergence_start:emergence_end]
    emergence_rmse = np.sqrt(np.mean((emergence_observed - emergence_predicted) ** 2))
    
    # Calculate overall RMSE
    overall_rmse = np.sqrt(np.mean((observed - predicted) ** 2))
    
    return {
        'emergence_rmse': emergence_rmse,
        'overall_rmse': overall_rmse
    }

def calculate_tile_level_emergence_metrics(observed: np.ndarray, predicted: np.ndarray, tile_indices: np.ndarray) -> Dict[str, float]:
    """Calculate emergence metrics aggregated across tiles."""
    unique_tiles = np.unique(tile_indices)
    tile_metrics = []
    
    for tile in unique_tiles:
        tile_mask = tile_indices == tile
        tile_observed = observed[tile_mask]
        tile_predicted = predicted[tile_mask]
        
        if len(tile_observed) > 0:
            metrics = calculate_emergence_metrics(tile_observed, tile_predicted)
            tile_metrics.append(metrics)
    
    if not tile_metrics:
        return {'emergence_rmse': np.nan, 'overall_rmse': np.nan}
    
    # Average metrics across tiles
    avg_emergence_rmse = np.nanmean([m['emergence_rmse'] for m in tile_metrics])
    avg_overall_rmse = np.nanmean([m['overall_rmse'] for m in tile_metrics])
    
    return {
        'emergence_rmse': avg_emergence_rmse,
        'overall_rmse': avg_overall_rmse
    }

def load_all_ars_data(ARs, rid_of_top, size, data_path_template):
    global GLOBAL_NORM_STATS
    if GLOBAL_NORM_STATS is not None:
        print("Using cached global normalization stats.")
    else:
        print("Computing global normalization stats...")
        all_stacked_maps, all_mag_flux, all_intensities_raw = [], [], []
        for AR in ARs:
            try:
                power_maps = np.load(data_path_template.format(AR=AR, type='pmdop', suffix='_flat.npz'), allow_pickle=True)
                mag_flux = np.load(data_path_template.format(AR=AR, type='mag', suffix='_flat.npz'), allow_pickle=True)
                intensities = np.load(data_path_template.format(AR=AR, type='int', suffix='_flat.npz'), allow_pickle=True)
                stacked_maps = np.stack([power_maps[f'arr_{i}'] for i in range(4)], axis=1)
                sl = slice(rid_of_top*size, -rid_of_top*size if rid_of_top > 0 else None)
                all_stacked_maps.append(np.nan_to_num(stacked_maps[sl, :]))
                all_mag_flux.append(np.nan_to_num(mag_flux['arr_0'][sl, :]))
                all_intensities_raw.append(np.nan_to_num(intensities['arr_0'][sl, :]))
            except FileNotFoundError:
                print(f"Warning: Data for AR {AR} not found. Skipping.")
                continue
        
        all_stacked_concat = np.concatenate(all_stacked_maps, axis=0)
        all_mag_concat = np.concatenate(all_mag_flux, axis=0)
        all_int_concat = np.concatenate(all_intensities_raw, axis=0)

        GLOBAL_NORM_STATS = {
            'min_p': np.min(all_stacked_concat), 'max_p': np.max(all_stacked_concat),
            'min_m': np.min(all_mag_concat), 'max_m': np.max(all_mag_concat),
            'min_i': np.min(all_int_concat), 'max_i': np.max(all_int_concat)
        }
        print("Global normalization stats computed and cached.")

    # Process and normalize data for all ARs
    all_inputs, all_intensities = [], []
    for AR in ARs:
        try:
            power_maps = np.load(data_path_template.format(AR=AR, type='pmdop', suffix='_flat.npz'), allow_pickle=True)
            mag_flux = np.load(data_path_template.format(AR=AR, type='mag', suffix='_flat.npz'), allow_pickle=True)
            intensities = np.load(data_path_template.format(AR=AR, type='int', suffix='_flat.npz'), allow_pickle=True)
            stacked_maps = np.stack([power_maps[f'arr_{i}'] for i in range(4)], axis=1)
            sl = slice(rid_of_top*size, -rid_of_top*size if rid_of_top > 0 else None)
            
            stacked_maps = np.nan_to_num(stacked_maps[sl, :])
            mag_flux = np.nan_to_num(mag_flux['arr_0'][sl, :])
            intensities = np.nan_to_num(intensities['arr_0'][sl, :])
            
            stacked_maps = (stacked_maps - GLOBAL_NORM_STATS['min_p']) / (GLOBAL_NORM_STATS['max_p'] - GLOBAL_NORM_STATS['min_p'])
            mag_flux = (mag_flux - GLOBAL_NORM_STATS['min_m']) / (GLOBAL_NORM_STATS['max_m'] - GLOBAL_NORM_STATS['min_m'])
            intensities = (intensities - GLOBAL_NORM_STATS['min_i']) / (GLOBAL_NORM_STATS['max_i'] - GLOBAL_NORM_STATS['min_i'])
            
            mag_flux_reshaped = np.expand_dims(mag_flux, axis=1)
            pm_and_flux = np.concatenate([stacked_maps, mag_flux_reshaped], axis=1)
            all_inputs.append(pm_and_flux)
            all_intensities.append(intensities)
        except FileNotFoundError:
            continue

    return np.stack(all_inputs, axis=-1), np.stack(all_intensities, axis=-1)

def lstm_ready(tile, size, power_maps, intensities, num_in, num_pred):
    """Use the same data preprocessing as LSTM code.
    
    Args:
        tile: Tile index
        size: Grid size
        power_maps: Power maps data
        intensities: Intensity data
        num_in: Requested input sequence length
        num_pred: Number of prediction steps
    """
    # Use the same preprocessing as LSTM code
    final_maps = np.transpose(power_maps, axes=(2, 1, 0))  # (time, features, tiles)
    final_ints = np.transpose(intensities, axes=(1,0))     # (time, tiles)
    X_trans = final_maps[:,:,tile]  # (time, features)
    y_trans = final_ints[:,tile]    # (time,)
    
    # Split into sequences
    X_ss, y_mm = split_sequences(X_trans, y_trans, num_in, num_pred)
    
    # Convert to tensors
    X = torch.Tensor(X_ss)  # (batch, seq_len, input_dim)
    y = torch.Tensor(y_mm)  # (batch, output_dim)
    
    return X, y

def main():
    import wandb
    run = wandb.init()
    config = wandb.config

    # Set global variables for force_ar_evaluation
    # global run_instance, model_save_path_global, config_global
    # run_instance = run
    # model_save_path_global = None # Will be set if best model is saved
    # config_global = config

    # Now safe to access config for run naming/notes
    run_name = f"sweep_conv_embed{getattr(config, 'embed_dim', 'NA')}_ff{getattr(config, 'ff_dim', 'NA')}_layers{getattr(config, 'num_layers', 'NA')}_drop{getattr(config, 'dropout', 'NA')}_lr{getattr(config, 'learning_rate', 'NA')}"
    run_notes = f"Sweep run with temporal convolution: embed_dim={getattr(config, 'embed_dim', 'NA')}, ff_dim={getattr(config, 'ff_dim', 'NA')}, num_layers={getattr(config, 'num_layers', 'NA')}, dropout={getattr(config, 'dropout', 'NA')}, lr={getattr(config, 'learning_rate', 'NA')}"
    run.name = run_name
    run.notes = run_notes

    print("--- Starting W&B Sweep Run with Temporal Convolution ---")
    print(f"Config: {config}")

    # Sanity check for hyperparameters
    if config.embed_dim % config.num_heads != 0:
        print(f"Skipping invalid run: embed_dim ({config.embed_dim}) is not divisible by num_heads ({config.num_heads}).")
        run.finish(exit_code=1)
        return

    # Log all hyperparameters at the start (clean, no duplicates)
    hyperparams = {
        'embed_dim': config.embed_dim,
        'ff_dim': config.ff_dim, 
        'num_layers': config.num_layers,
        'num_heads': config.num_heads,
        'dropout': config.dropout,
        'learning_rate': config.learning_rate,
        'n_epochs': config.n_epochs,
        'num_in': config.num_in,
        'num_pred': config.num_pred,
        'rid_of_top': config.rid_of_top,
        'time_window': config.time_window
    }
    wandb.log(hyperparams)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model with temporal convolution
    model = SpatioTemporalTransformerConv(
        input_dim=5,  # Fixed: 5 features (4 power maps + 1 magnetic flux)
        seq_len=config.num_in,
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        ff_dim=config.ff_dim,
        num_layers=config.num_layers,
        output_dim=config.num_pred,
        dropout=config.dropout
    ).to(device)
    
    # IMMEDIATELY save the initialized model to ensure we always have a model for evaluation
    # This guarantees AR evaluation works even for stopped runs
    model_save_dir = '/project/mx6/jst26/SAR_EMERGENCE_RESEARCH/transformer/models/sweep #2'
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Create unique identifiers for this run
    timestamp = int(time.time())
    run_name_safe = wandb.run.name.replace(' ', '_').replace('/', '_') if wandb.run.name else f"run_{wandb.run.id}"
    initial_model_path = os.path.join(model_save_dir, f"conv_initial_{run_name_safe}_{wandb.run.id}_{timestamp}.pth")
    torch.save(model.state_dict(), initial_model_path)
    print(f"Initial model saved for guaranteed evaluation: {initial_model_path}")
    
    # Also save as artifact immediately
    artifact = wandb.Artifact(
        name=f"initial_model_{wandb.run.id}", 
        type="model",
        description="Initial model state for guaranteed AR evaluation"
    )
    artifact.add_file(initial_model_path)
    wandb.log_artifact(artifact)
    print("Initial model uploaded as artifact for guaranteed evaluation")

    # Variables to track training state
    model_save_path = None
    best_model_state = None
    best_epoch = 0
    best_test_loss = float('inf')
    
    try:
        # Data Loading
        ARs = [11130, 11149, 11158, 11162, 11199, 11327, 11344, 11387, 11393, 11416, 11422, 11455, 11619, 11640, 11660, 11678, 11682, 11765, 11768, 11776, 11916, 11928, 12036, 12051, 12085, 12089, 12144, 12175, 12203, 12257, 12331, 12494, 12659, 12778, 12864, 12877, 12900, 12929, 13004, 13085, 13098, 13179]
        size = 9
        data_path_template = '/mmfs1/project/mx6/jst26/SAR_EMERGENCE_RESEARCH/data/AR{AR}/mean_{type}{AR}{suffix}'
        
        print("Loading all ARs data...")
        try:
            set_timeout(300)  # 5 minute timeout for data loading
            all_inputs, all_intensities = load_all_ars_data(ARs, config.rid_of_top, size, data_path_template)
            signal.alarm(0)  # Cancel timeout
        except TimeoutError:
            print("Error: Data loading timed out after 5 minutes. Proceeding to AR evaluation with initial model.")
            return  # This will trigger the finally block
        except Exception as e:
            print(f"Error during data loading: {e}. Proceeding to AR evaluation with initial model.")
            return  # This will trigger the finally block
        
        print(f"Data loaded. Shape: inputs={all_inputs.shape}, intensities={all_intensities.shape}")
        
        print("Processing data for training...")
        X_trains, y_trains, tile_indices = [], [], []
        tiles = all_inputs.shape[0]
        total_ars = all_inputs.shape[-1]
        
        print(f"Processing {total_ars} ARs with {tiles} tiles each...")
        try:
            set_timeout(600)  # 10 minute timeout for data processing
            max_samples = 100000  # Limit total samples to prevent memory issues
            samples_created = 0
            
            for ar_idx in range(total_ars):
                if ar_idx % 5 == 0:  # Progress tracking every 5 ARs
                    print(f"Processing AR {ar_idx+1}/{total_ars}")
                
                power_maps = all_inputs[:, :, :, ar_idx]
                intensities = all_intensities[:, :, ar_idx]
                
                # Process tiles in batches to avoid memory issues
                for tile in range(tiles):
                    if samples_created >= max_samples:
                        print(f"Reached maximum samples limit ({max_samples}). Stopping data processing.")
                        break
                        
                    try:
                        X_tile, y_tile = lstm_ready(tile, size, power_maps, intensities, config.num_in, config.num_pred)
                        if len(X_tile) > 0:  # Only add if we have data
                            X_trains.append(X_tile)
                            y_trains.append(y_tile)
                            tile_indices.extend([tile] * len(X_tile))
                            samples_created += len(X_tile)
                    except Exception as e:
                        print(f"Warning: Error processing tile {tile} for AR {ar_idx}: {e}")
                        continue
                
                # Garbage collection every 5 ARs to prevent memory buildup
                if ar_idx % 5 == 0:
                    gc.collect()
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
                if samples_created >= max_samples:
                    break
                    
            signal.alarm(0)  # Cancel timeout
        except TimeoutError:
            print("Error: Data processing timed out after 10 minutes. Proceeding to AR evaluation with initial model.")
            return  # This will trigger the finally block
        except Exception as e:
            print(f"Error during data processing: {e}. Proceeding to AR evaluation with initial model.")
            return  # This will trigger the finally block

        if len(X_trains) == 0:
            print("Error: No training data created. Proceeding to AR evaluation with initial model.")
            return  # This will trigger the finally block

        # Concatenate all training data
        X_train = torch.cat(X_trains, dim=0)
        y_train = torch.cat(y_trains, dim=0)
        tile_indices = np.array(tile_indices)
        
        print(f"Data processing complete. Created {len(X_trains)} training samples ({len(X_train)} total samples).")
        print(f"Final data shapes: X={X_train.shape}, y={y_train.shape}, tile_indices={tile_indices.shape}")

        # Create train/test split
        total_samples = len(X_train)
        train_size = int(0.8 * total_samples)
        indices = torch.randperm(total_samples)
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        
        X_train_split = X_train[train_indices]
        y_train_split = y_train[train_indices]
        X_test = X_train[test_indices]
        y_test = y_train[test_indices]
        
        # Create DataLoader
        train_dataset = TensorDataset(X_train_split, y_train_split)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        
        # Training setup
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
        loss_fn = nn.MSELoss()
        
        # Learning rate scheduler
        total_steps = len(train_loader) * config.n_epochs
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=total_steps//10, num_training_steps=total_steps)

        try:
            # Training Loop
            print(f"Starting training for {config.n_epochs} epochs...")
            try:
                set_timeout(3600)  # 1 hour timeout for training
                for epoch in range(config.n_epochs):
                    if epoch % 10 == 0:  # Progress tracking every 10 epochs
                        print(f"Epoch {epoch+1}/{config.n_epochs}")
                    
                    model.train()
                    epoch_train_loss = 0
                    for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
                        if batch_idx == 0 and epoch == 0:  # Debug first batch of first epoch
                            print(f"Debug - batch_X shape: {batch_X.shape}, batch_y shape: {batch_y.shape}")
                            print(f"Debug - model seq_len: {config.num_in}")
                        
                        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                        optimizer.zero_grad()
                        outputs = model(batch_X)
                        loss = loss_fn(outputs, batch_y)
                        if torch.isnan(loss): continue
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        epoch_train_loss += loss.item()
                    
                    epoch_train_loss /= len(train_loader)

                    model.eval()
                    epoch_test_loss = 0
                    all_test_preds, all_test_targets = [], []
                    with torch.no_grad():
                        for batch_X, batch_y in train_loader:  # Using train_loader for simplicity
                            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                            outputs = model(batch_X)
                            loss = loss_fn(outputs, batch_y)
                            epoch_test_loss += loss.item()
                            all_test_preds.append(outputs.cpu().numpy())
                            all_test_targets.append(batch_y.cpu().numpy())
                    
                    epoch_test_loss /= len(train_loader)
                    
                    # Calculate emergence metrics
                    all_test_preds = np.concatenate(all_test_preds, axis=0)
                    all_test_targets = np.concatenate(all_test_targets, axis=0)
                    test_emergence_metrics = calculate_emergence_metrics(all_test_targets.flatten(), all_test_preds.flatten())
                    
                    # Save best model (only every 10 epochs to reduce file count)
                    if epoch_test_loss < best_test_loss:
                        best_test_loss = epoch_test_loss
                        best_epoch = epoch
                        best_model_state = model.state_dict().copy()
                        
                        # Only save model file every 10 epochs or on final epoch
                        if epoch % 10 == 0 or epoch == config.n_epochs - 1:
                            # Delete previous model file if it exists
                            if model_save_path and os.path.exists(model_save_path):
                                try:
                                    os.remove(model_save_path)
                                    print(f"Deleted previous model: {model_save_path}")
                                except Exception as e:
                                    print(f"Warning: Could not delete {model_save_path}: {e}")
                            
                            model_save_path = os.path.join(model_save_dir, f"conv_best_{run_name_safe}_{wandb.run.id}_epoch{epoch}_loss{epoch_test_loss:.4f}_{timestamp}.pth")
                            torch.save(best_model_state, model_save_path)
                            print(f"New best model saved: {model_save_path}")
                        else:
                            # Keep track of best model state without saving file
                            model_save_path = None
                    
                    # Log metrics
                    wandb.log({
                        'train_loss': epoch_train_loss,
                        'test_loss': epoch_test_loss,
                        'test_emergence_rmse': test_emergence_metrics.get('emergence_rmse'),
                        'test_overall_rmse': test_emergence_metrics.get('overall_rmse'),
                        'epoch': epoch,
                        'learning_rate': optimizer.param_groups[0]['lr']
                    })
                    scheduler.step()
                
                signal.alarm(0)  # Cancel timeout
            except TimeoutError:
                print("Error: Training timed out after 1 hour. Proceeding to AR evaluation with current model.")
            except Exception as e:
                print(f"Error during training: {e}")
                # Continue to AR evaluation with whatever model we have

            print("--- W&B Sweep Run Finished ---")
            # Save and upload the best model as a wandb artifact
            if best_model_state is not None:
                # Always save the final best model for evaluation
                final_model_path = f"best_model_epoch{best_epoch}_loss{best_test_loss:.4f}.pth"
                torch.save(best_model_state, final_model_path)
                
                # Clean up any intermediate model files
                if model_save_path and model_save_path != final_model_path and os.path.exists(model_save_path):
                    try:
                        os.remove(model_save_path)
                        print(f"Cleaned up intermediate model: {model_save_path}")
                    except Exception as e:
                        print(f"Warning: Could not delete {model_save_path}: {e}")
                
                artifact = wandb.Artifact(
                    name=f"best_model_{wandb.run.id}",
                    type="model",
                    description=f"Best model for sweep run (epoch {best_epoch}, test_loss {best_test_loss:.4f})",
                    metadata={
                        "best_epoch": best_epoch,
                        "best_test_loss": best_test_loss,
                        "config": dict(config)
                    }
                )
                artifact.add_file(final_model_path)
                wandb.log_artifact(artifact)
                print(f"Best model saved and uploaded as artifact: {final_model_path}")
                
                # Update model_save_path for AR evaluation
                model_save_path = final_model_path
        except Exception as e:
            print(f"Error during training setup or execution: {e}")
            # Continue to AR evaluation with initial model
    finally:
        # AR evaluation and artifact upload - ALWAYS RUNS
        print("Starting AR evaluation...")
        
        try:
            set_timeout(1800)  # 30 minute timeout for AR evaluation
            
            # Ensure we have a model path for evaluation
            if model_save_path is None:
                # Use the initial model that was saved immediately after initialization
                model_save_path = os.path.join(model_save_dir, f"conv_initial_{run_name_safe}_{wandb.run.id}_{timestamp}.pth")
                print(f"No best model found during training, using initial model: {model_save_path}")
                if not os.path.exists(model_save_path):
                    print("Error: Initial model not found. Cannot perform AR evaluation.")
                    run.finish()
                    return

            # Define an output directory for the evaluation plots inside the W&B run folder
            plot_output_dir = os.path.join(wandb.run.dir, "ar_comparison_plots")
            os.makedirs(plot_output_dir, exist_ok=True)
            print(f"Created plot output directory: {plot_output_dir}")
            
            # AR evaluation
            test_ars = [11698, 11726, 13165, 13179, 13183]
            lstm_path = "/mmfs1/project/mx6/jst26/SAR_EMERGENCE_RESEARCH/lstm/results/t12_r4_i110_n3_h64_e1000_l0.01.pth"
            print(f"Starting AR evaluation for {len(test_ars)} ARs: {test_ars}")
            print(f"Using transformer model: {model_save_path}")
            print(f"Model file exists: {os.path.exists(model_save_path)}")
            print(f"Model file size: {os.path.getsize(model_save_path) if os.path.exists(model_save_path) else 'N/A'} bytes")
            
            # Debug: Check the saved model structure
            if os.path.exists(model_save_path):
                try:
                    saved_state = torch.load(model_save_path, map_location='cpu')
                    print(f"Debug - Saved model keys: {list(saved_state.keys())}")
                    if 'embedding.weight' in saved_state:
                        print(f"Debug - Saved embedding.weight shape: {saved_state['embedding.weight'].shape}")
                except Exception as e:
                    print(f"Debug - Error loading saved model: {e}")
            
            print(f"Using LSTM model: {lstm_path}")
            print(f"Output directory: {plot_output_dir}")
            print(f"Active W&B run: {run.name} (ID: {run.id})")
            
            plots_logged = []
            for i, ar in enumerate(test_ars):
                try:
                    print(f"Evaluating AR {ar} ({i+1}/{len(test_ars)})...")
                    
                    # Create transformer_params dict for evaluation
                    transformer_params = {
                        'embed_dim': config.embed_dim,
                        'num_heads': config.num_heads,
                        'ff_dim': config.ff_dim,
                        'num_layers': config.num_layers,
                        'dropout': config.dropout,
                        'rid_of_top': config.rid_of_top,
                        'num_pred': config.num_pred,
                        'time_window': config.time_window,
                        'num_in': config.num_in,
                        'input_dim': 5,  # Fixed: 5 features (4 power maps + 1 magnetic flux)
                        'hidden_size': config.embed_dim,
                        'learning_rate': config.learning_rate
                    }
                    
                    print(f"Debug - transformer_params for AR {ar}: {transformer_params}")
                    
                    # CORRECTED FUNCTION CALL
                    # Signature: evaluate_models_for_ar(test_AR, lstm_path, transformer_path, transformer_params, output_dir)
                    plot_path = evaluate_models_for_ar(
                         ar,                 # test_AR
                         lstm_path,          # lstm_path
                         model_save_path,    # transformer_path
                         transformer_params, # transformer_params
                         plot_output_dir     # output_dir
                    )
                     
                    if plot_path and os.path.exists(plot_path):
                        print(f"AR {ar} evaluation successful, logging plot: {plot_path}")
                        # Log to individual run with run-specific naming
                        wandb.log({
                            f'AR_{ar}_comparison': wandb.Image(
                                plot_path,
                                caption=f"AR {ar} comparison for run: {run.name}"
                            )
                        })
                        plots_logged.append(f"AR_{ar}_comparison")
                        print(f"Successfully logged AR {ar} plot to run: {run.name}")
                    else:
                        print(f"Warning: No plot generated or found for AR {ar}")
                        
                except Exception as e:
                    import traceback
                    print(f"An unexpected error occurred while evaluating AR {ar}: {e}")
                    traceback.print_exc()  # Print full traceback for easier debugging
                    continue
            
            signal.alarm(0)  # Cancel timeout
            
            # Final summary
            print(f"AR evaluation complete. Plots logged to run {run.name}: {plots_logged}")
            print(f"Total plots logged: {len(plots_logged)} out of {len(test_ars)} ARs")
            
        except TimeoutError:
            print("Error: AR evaluation timed out after 30 minutes.")
        except Exception as e:
            print(f"Error during AR evaluation: {e}")
        
        run.finish()

if __name__ == "__main__":
    main() 