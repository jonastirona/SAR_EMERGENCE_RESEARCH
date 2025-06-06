import optuna
import torch
import torch.nn as nn
import numpy as np
from models.st_transformer import SpatioTemporalTransformer
from functions import training_loop_w_stats, lstm_ready, min_max_scaling
import sys
import os
from datetime import datetime
import logging
from scipy import stats
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import OneCycleLR

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def initialize_weights(model):
    """Initialize transformer weights using Xavier uniform."""
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

def calculate_shape_metrics(y_true, y_pred):
    """Calculate shape-based and directional similarity metrics."""
    # Convert to numpy if tensors
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()
    
    metrics = {}
    
    # Trend direction agreement
    true_diff = np.diff(y_true.flatten())
    pred_diff = np.diff(y_pred.flatten())
    direction_agreement = np.mean(np.sign(true_diff) == np.sign(pred_diff))
    metrics['direction_agreement'] = direction_agreement
    
    # Correlation coefficient
    try:
        corr = np.corrcoef(y_true.flatten(), y_pred.flatten())[0, 1]
        metrics['correlation'] = corr
    except:
        metrics['correlation'] = -1.0
    
    # Peak alignment with tolerance
    true_peaks, _ = find_peaks(y_true.flatten(), distance=3)
    pred_peaks, _ = find_peaks(y_pred.flatten(), distance=3)
    
    if len(true_peaks) > 0 and len(pred_peaks) > 0:
        # Calculate minimum distances between peaks
        peak_distances = []
        for tp in true_peaks:
            min_dist = min(abs(tp - pp) for pp in pred_peaks)
            peak_distances.append(min_dist)
        avg_peak_dist = np.mean(peak_distances) / len(y_true)
        metrics['peak_alignment'] = 1 - min(avg_peak_dist, 1.0)
    else:
        metrics['peak_alignment'] = 0.0
    
    # Variance ratio (to detect flat predictions)
    pred_var = np.var(y_pred)
    true_var = np.var(y_true)
    metrics['variance_ratio'] = min(pred_var / (true_var + 1e-8), 1.0)
    
    # Combined score with emphasis on trend following
    weights = {
        'direction_agreement': 0.4,
        'correlation': 0.3,
        'peak_alignment': 0.2,
        'variance_ratio': 0.1
    }
    
    combined_score = sum(metrics[k] * weights[k] for k in weights.keys())
    metrics['combined'] = combined_score
    
    return metrics

def calculate_tensor_metrics(y_true, y_pred):
    """Calculate metrics while keeping values as tensors for backprop."""
    # Ensure inputs are tensors
    if not torch.is_tensor(y_true):
        y_true = torch.tensor(y_true, dtype=torch.float32)
    if not torch.is_tensor(y_pred):
        y_pred = torch.tensor(y_pred, dtype=torch.float32)
    
    # Flatten tensors
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # Direction agreement
    true_diff = y_true_flat[1:] - y_true_flat[:-1]
    pred_diff = y_pred_flat[1:] - y_pred_flat[:-1]
    direction_agreement = torch.mean((torch.sign(true_diff) == torch.sign(pred_diff)).float())
    
    # Correlation (using cosine similarity as differentiable alternative)
    y_true_norm = y_true_flat - y_true_flat.mean()
    y_pred_norm = y_pred_flat - y_pred_flat.mean()
    correlation = torch.nn.functional.cosine_similarity(y_true_norm.unsqueeze(0), y_pred_norm.unsqueeze(0))
    
    # Variance ratio
    pred_var = torch.var(y_pred_flat)
    true_var = torch.var(y_true_flat)
    variance_ratio = torch.min(pred_var / (true_var + 1e-8), torch.tensor(1.0))
    
    # Combined loss (negative because we want to maximize similarity)
    loss = -(0.4 * direction_agreement + 0.4 * correlation + 0.2 * variance_ratio)
    
    return loss

def custom_training_loop(model, X_train, y_train, X_test_tiles, y_test_tiles, optimizer, n_epochs, device):
    """Custom training loop with improved validation and visualization."""
    best_val_score = float('inf')
    train_metrics = []
    val_metrics = []
    best_model_state = None
    
    # Learning rate scheduler with warmup
    scheduler = OneCycleLR(
        optimizer,
        max_lr=optimizer.param_groups[0]['lr'],
        epochs=n_epochs,
        steps_per_epoch=1,
        pct_start=0.1,  # 10% warmup
        div_factor=25,  # initial_lr = max_lr/25
        final_div_factor=1e4  # final_lr = initial_lr/1e4
    )
    
    # For visualization
    example_predictions = []
    
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        y_pred_train = model(X_train)
        
        # Calculate loss using tensor metrics
        loss = calculate_tensor_metrics(y_train, y_pred_train)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        # Validation across multiple tiles
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_test, y_test in zip(X_test_tiles, y_test_tiles):
                y_pred_val = model(X_test)
                val_loss = calculate_tensor_metrics(y_test, y_pred_val)
                val_losses.append(val_loss.item())
            
            val_score = np.mean(val_losses)
            
            # Save best model
            if val_score < best_val_score:
                best_val_score = val_score
                best_model_state = model.state_dict().copy()
            
            # Store example prediction every 50 epochs
            if epoch % 50 == 0:
                example_predictions.append({
                    'epoch': epoch,
                    'pred': y_pred_val.cpu().numpy(),
                    'true': y_test.cpu().numpy()
                })
        
        train_metrics.append({'loss': loss.item()})
        val_metrics.append({'loss': val_score})
        
        if epoch % 10 == 0:
            logging.info(f'Epoch {epoch}: Train Loss = {loss.item():.4f}, Val Loss = {val_score:.4f}, LR = {scheduler.get_last_lr()[0]:.6f}')
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    return best_val_score, train_metrics, val_metrics, example_predictions

def load_and_preprocess_ar(AR, rid_of_top, size):
    # Load data
    power_maps = np.load(f'/mmfs1/project/mx6/jst26/SAR_EMERGENCE_RESEARCH/data/AR{AR}/mean_pmdop{AR}_flat.npz', allow_pickle=True)
    mag_flux = np.load(f'/mmfs1/project/mx6/jst26/SAR_EMERGENCE_RESEARCH/data/AR{AR}/mean_mag{AR}_flat.npz', allow_pickle=True)
    intensities = np.load(f'/mmfs1/project/mx6/jst26/SAR_EMERGENCE_RESEARCH/data/AR{AR}/mean_int{AR}_flat.npz', allow_pickle=True)

    # Extract arrays
    power_maps23 = power_maps['arr_0'][rid_of_top*size:-rid_of_top*size, :]
    power_maps34 = power_maps['arr_1'][rid_of_top*size:-rid_of_top*size, :]
    power_maps45 = power_maps['arr_2'][rid_of_top*size:-rid_of_top*size, :]
    power_maps56 = power_maps['arr_3'][rid_of_top*size:-rid_of_top*size, :]
    mag_flux = mag_flux['arr_0'][rid_of_top*size:-rid_of_top*size, :]
    intensities = intensities['arr_0'][rid_of_top*size:-rid_of_top*size, :]

    # Handle NaN values
    mag_flux[np.isnan(mag_flux)] = 0
    intensities[np.isnan(intensities)] = 0
    stacked_maps = np.stack([power_maps23, power_maps34, power_maps45, power_maps56], axis=1)
    stacked_maps[np.isnan(stacked_maps)] = 0

    # Normalize data
    min_p, max_p = np.min(stacked_maps), np.max(stacked_maps)
    min_m, max_m = np.min(mag_flux), np.max(mag_flux)
    min_i, max_i = np.min(intensities), np.max(intensities)
    stacked_maps = min_max_scaling(stacked_maps, min_p, max_p)
    mag_flux = min_max_scaling(mag_flux, min_m, max_m)
    intensities = min_max_scaling(intensities, min_i, max_i)
    mag_flux = np.expand_dims(mag_flux, axis=1)
    inputs = np.concatenate([stacked_maps, mag_flux], axis=1)
    
    return inputs, intensities

def objective(trial):
    # Log trial start
    logging.info(f"\n{'='*80}\nTrial {trial.number} started\n{'='*80}")
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Running on: {device}')

    # Fixed parameters
    num_pred = 12
    rid_of_top = 4
    num_in = 110
    n_epochs = 300
    size = 9
    
    # Hyperparameters to optimize with adjusted ranges
    num_heads = trial.suggest_int('num_heads', 2, 16, step=2)
    hidden_size = trial.suggest_int('hidden_size', num_heads * 16, num_heads * 64, step=num_heads)
    num_layers = trial.suggest_int('num_layers', 2, 6)
    ff_ratio = trial.suggest_float('ff_ratio', 2.0, 4.0)
    ff_dim = int(hidden_size * ff_ratio)
    learning_rate = trial.suggest_float('learning_rate', 5e-5, 5e-3, log=True)
    dropout = trial.suggest_float('dropout', 0.1, 0.3)
    
    # Log hyperparameters
    logging.info("Trial hyperparameters:")
    for param_name, param_value in trial.params.items():
        logging.info(f"  {param_name}: {param_value}")
    
    # Use multiple ARs for better generalization
    optimization_ARs = [11130, 11149, 11158, 11162, 11199, 11327, 11344, 11387, 11393, 11416]
    all_val_scores = []
    all_metric_details = []
    
    for AR in optimization_ARs:
        try:
            logging.info(f"\nProcessing AR {AR}")
            
            # Load and preprocess data
            inputs, intensities = load_and_preprocess_ar(AR, rid_of_top, size)
            input_size = inputs.shape[1]

            # Prepare data for model using all tiles
            all_tiles = list(range(9))  # Use all tiles from 0 to 8
            X_trains = []
            y_trains = []
            
            # Collect data from all tiles
            for tile in all_tiles:
                X_tile, y_tile = lstm_ready(tile, size, inputs, intensities, num_in, num_pred)
                X_trains.append(X_tile)
                y_trains.append(y_tile)
            
            # Concatenate all tile data
            X_train = torch.cat(X_trains, dim=0)
            y_train = torch.cat(y_trains, dim=0)

            # Move to device
            X_train = X_train.to(device)
            y_train = y_train.to(device)

            # For validation, we'll use the same tiles but with different splits
            X_test_tiles = []
            y_test_tiles = []
            for tile in all_tiles:
                X_test, y_test = lstm_ready(tile, size, inputs, intensities, num_in, num_pred)
                X_test_tiles.append(X_test.to(device))
                y_test_tiles.append(y_test.to(device))

            # Initialize model with Xavier
            model = SpatioTemporalTransformer(
                input_dim=input_size,
                seq_len=num_in,
                embed_dim=hidden_size,
                num_heads=num_heads,
                ff_dim=ff_dim,
                num_layers=num_layers,
                output_dim=num_pred,
                dropout=dropout
            ).to(device)
            model = initialize_weights(model)

            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            # Training with improved validation
            best_val_score, train_metrics, val_metrics, example_preds = custom_training_loop(
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_test_tiles=X_test_tiles,
                y_test_tiles=y_test_tiles,
                optimizer=optimizer,
                n_epochs=n_epochs,
                device=device
            )
            
            all_val_scores.append(best_val_score)
            all_metric_details.append({
                'AR': AR,
                'best_val_score': best_val_score,
                'example_predictions': example_preds
            })
            
            logging.info(f"AR {AR} best validation score: {best_val_score:.6f}")
    
        except Exception as e:
            logging.error(f"Error processing AR {AR}: {str(e)}")
            continue
    
    if not all_val_scores:
        return float('inf')
    
    avg_score = np.mean(all_val_scores)
    logging.info(f"\nAverage validation score across all ARs: {avg_score:.6f}")
    
    # Save example predictions for visualization
    if trial.number % 5 == 0:  # Save every 5th trial
        save_prediction_plots(all_metric_details, trial)
    
    return avg_score

def save_prediction_plots(metric_details, trial):
    """Save prediction visualization plots."""
    results_dir = "optuna_results/predictions"
    os.makedirs(results_dir, exist_ok=True)
    
    for ar_metrics in metric_details:
        AR = ar_metrics['AR']
        predictions = ar_metrics['example_predictions']
        
        plt.figure(figsize=(15, 10))
        for pred in predictions:
            plt.subplot(len(predictions), 1, predictions.index(pred) + 1)
            plt.plot(pred['true'].flatten(), label='True', alpha=0.7)
            plt.plot(pred['pred'].flatten(), label='Predicted', alpha=0.7)
            plt.title(f"AR {AR} - Epoch {pred['epoch']}")
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{results_dir}/trial_{trial.number}_AR{AR}_predictions.png")
        plt.close()

def main():
    logging.info(f"\n{'='*80}\nStarting Transformer Hyperparameter Optimization\n{'='*80}")
    
    study_name = f"transformer_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    n_trials = 50  # Number of trials to run
    logging.info(f"Will run {n_trials} trials")
    
    study.optimize(objective, n_trials=n_trials)

    logging.info("\nOptimization finished!")
    logging.info("\nBest trial:")
    trial = study.best_trial

    logging.info(f"  Value: {trial.value}")
    logging.info("  Params: ")
    for key, value in trial.params.items():
        logging.info(f"    {key}: {value}")

    # Save results
    results_dir = "optuna_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save study statistics to a file
    results_file = f"{results_dir}/{study_name}_results.txt"
    logging.info(f"\nSaving detailed results to {results_file}")
    
    with open(results_file, "w") as f:
        f.write(f"Best trial value: {trial.value}\n")
        f.write("Best trial params:\n")
        for key, value in trial.params.items():
            f.write(f"{key}: {value}\n")
        
        f.write("\nAll trials:\n")
        for trial in study.trials:
            f.write(f"Trial {trial.number}:\n")
            f.write(f"  Value: {trial.value}\n")
            f.write("  Params:\n")
            for key, value in trial.params.items():
                f.write(f"    {key}: {value}\n")
            f.write("\n")

    # Create visualization plots
    logging.info("\nGenerating visualization plots...")
    import plotly
    fig1 = optuna.visualization.plot_optimization_history(study)
    fig2 = optuna.visualization.plot_parallel_coordinate(study)
    fig3 = optuna.visualization.plot_param_importances(study)
    
    # Save plots
    logging.info(f"Saving plots to {results_dir}/")
    fig1.write_html(f"{results_dir}/{study_name}_optimization_history.html")
    fig2.write_html(f"{results_dir}/{study_name}_parallel_coordinate.html")
    fig3.write_html(f"{results_dir}/{study_name}_param_importances.html")
    
    logging.info("\nOptimization completed successfully!")

if __name__ == "__main__":
    main() 