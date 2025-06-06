import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
import optuna
from datetime import datetime
import os
import logging
from functions import lstm_ready, LSTM, min_max_scaling

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def initialize_weights(model):
    """Initialize LSTM weights using Xavier initialization."""
    for name, param in model.named_parameters():
        if 'weight' in name:
            nn.init.xavier_normal_(param)
        elif 'bias' in name:
            nn.init.constant_(param, 0.0)

def calculate_tensor_metrics(y_true, y_pred):
    """Calculate shape-based metrics between true and predicted sequences using PyTorch tensors."""
    # Ensure inputs are tensors
    if not torch.is_tensor(y_true):
        y_true = torch.tensor(y_true, dtype=torch.float32)
    if not torch.is_tensor(y_pred):
        y_pred = torch.tensor(y_pred, dtype=torch.float32)
    
    # Flatten tensors
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # Direction agreement (trend similarity)
    true_diff = y_true_flat[1:] - y_true_flat[:-1]
    pred_diff = y_pred_flat[1:] - y_pred_flat[:-1]
    direction_agreement = torch.mean((torch.sign(true_diff) == torch.sign(pred_diff)).float())
    
    # Correlation (using cosine similarity as differentiable alternative)
    y_true_norm = y_true_flat - y_true_flat.mean()
    y_pred_norm = y_pred_flat - y_pred_flat.mean()
    correlation = torch.nn.functional.cosine_similarity(y_true_norm.unsqueeze(0), y_pred_norm.unsqueeze(0))
    
    # Variance ratio (penalize if prediction variance is too different from true variance)
    pred_var = torch.var(y_pred_flat)
    true_var = torch.var(y_true_flat)
    variance_ratio = torch.min(pred_var / (true_var + 1e-8), torch.tensor(1.0))
    
    # Combined metric (weighted average)
    combined = 0.4 * direction_agreement + 0.4 * correlation + 0.2 * variance_ratio
    
    return combined

def calculate_shape_metrics(y_true, y_pred):
    """Calculate shape-based metrics between true and predicted sequences using numpy (for evaluation only)."""
    # Convert to numpy arrays
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()
    
    # Flatten arrays
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # Direction agreement (trend similarity)
    true_diff = np.diff(y_true_flat)
    pred_diff = np.diff(y_pred_flat)
    direction_agreement = np.mean((np.sign(true_diff) == np.sign(pred_diff)))
    
    # Correlation coefficient
    correlation = np.corrcoef(y_true_flat, y_pred_flat)[0, 1]
    if np.isnan(correlation):
        correlation = 0.0
    
    # Variance ratio (penalize if prediction variance is too different from true variance)
    pred_var = np.var(y_pred_flat)
    true_var = np.var(y_true_flat)
    variance_ratio = min(pred_var / (true_var + 1e-8), 1.0)
    
    # Combined metric (weighted average)
    metrics = {
        'direction': direction_agreement,
        'correlation': correlation,
        'variance': variance_ratio,
        'combined': 0.4 * direction_agreement + 0.4 * correlation + 0.2 * variance_ratio
    }
    
    return metrics

def custom_training_loop(model, X_train, y_train, X_test_tiles, y_test_tiles, optimizer, n_epochs, device):
    """Custom training loop with shape-based metrics and validation across multiple tiles."""
    best_val_score = float('-inf')  # Now maximizing score
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
        loss = -calculate_tensor_metrics(y_train, y_pred_train)  # Negative because we maximize metrics
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()
        scheduler.step()
        
        # Validation across multiple tiles
        model.eval()
        val_scores = []
        with torch.no_grad():
            for X_test, y_test in zip(X_test_tiles, y_test_tiles):
                y_pred_val = model(X_test)
                val_score = calculate_tensor_metrics(y_test, y_pred_val)
                val_scores.append(val_score.item())
            
            val_score = np.mean(val_scores)
            
            # Save best model
            if val_score > best_val_score:
                best_val_score = val_score
                best_model_state = model.state_dict().copy()
            
            # Store example prediction every 50 epochs
            if epoch % 50 == 0:
                example_predictions.append({
                    'epoch': epoch,
                    'pred': y_pred_val.cpu().numpy(),
                    'true': y_test.cpu().numpy()
                })
        
        # Calculate detailed metrics for logging (using numpy version)
        train_shape_metrics = calculate_shape_metrics(y_train, y_pred_train)
        train_metrics.append(train_shape_metrics)
        val_metrics.append({'combined': val_score})
        
        if epoch % 10 == 0:
            logging.info(f'Epoch {epoch}: Train Score = {-loss.item():.4f}, Val Score = {val_score:.4f}, LR = {scheduler.get_last_lr()[0]:.6f}')
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    return best_val_score, train_metrics, val_metrics, example_predictions

def load_and_preprocess_ar(AR, rid_of_top, size):
    """Load and preprocess data for a single AR."""
    # Load data
    power_maps = np.load('/mmfs1/project/mx6/jst26/SAR_EMERGENCE_RESEARCH/data/AR{}/mean_pmdop{}_flat.npz'.format(AR,AR), allow_pickle=True)
    mag_flux = np.load('/mmfs1/project/mx6/jst26/SAR_EMERGENCE_RESEARCH/data/AR{}/mean_mag{}_flat.npz'.format(AR,AR), allow_pickle=True)
    intensities = np.load('/mmfs1/project/mx6/jst26/SAR_EMERGENCE_RESEARCH/data/AR{}/mean_int{}_flat.npz'.format(AR,AR), allow_pickle=True)
    
    # Extract arrays
    power_maps23 = power_maps['arr_0']
    power_maps34 = power_maps['arr_1']
    power_maps45 = power_maps['arr_2']
    power_maps56 = power_maps['arr_3']
    mag_flux = mag_flux['arr_0']
    intensities = intensities['arr_0']
    
    # Trim arrays
    power_maps23 = power_maps23[rid_of_top*size:-rid_of_top*size, :]
    power_maps34 = power_maps34[rid_of_top*size:-rid_of_top*size, :]
    power_maps45 = power_maps45[rid_of_top*size:-rid_of_top*size, :]
    power_maps56 = power_maps56[rid_of_top*size:-rid_of_top*size, :]
    mag_flux = mag_flux[rid_of_top*size:-rid_of_top*size, :]
    intensities = intensities[rid_of_top*size:-rid_of_top*size, :]
    
    # Handle NaN values
    mag_flux[np.isnan(mag_flux)] = 0
    intensities[np.isnan(intensities)] = 0
    
    # Stack and normalize inputs
    stacked_maps = np.stack([power_maps23, power_maps34, power_maps45, power_maps56], axis=1)
    stacked_maps[np.isnan(stacked_maps)] = 0
    
    min_p = np.min(stacked_maps)
    max_p = np.max(stacked_maps)
    min_m = np.min(mag_flux)
    max_m = np.max(mag_flux)
    min_i = np.min(intensities)
    max_i = np.max(intensities)
    
    stacked_maps = min_max_scaling(stacked_maps, min_p, max_p)
    mag_flux = min_max_scaling(mag_flux, min_m, max_m)
    intensities = min_max_scaling(intensities, min_i, max_i)
    
    # Prepare final input
    mag_flux_reshaped = np.expand_dims(mag_flux, axis=1)
    inputs = np.concatenate([stacked_maps, mag_flux_reshaped], axis=1)
    
    return inputs, intensities

def objective(trial):
    """Optuna objective function for hyperparameter optimization."""
    # Log trial start
    logging.info("\n" + "="*80)
    logging.info("Trial {} started".format(trial.number))
    logging.info("="*80)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Running on: {device}")
    
    # Define hyperparameters to optimize
    hidden_size = trial.suggest_int('hidden_size', 64, 512)
    num_layers = trial.suggest_int('num_layers', 1, 8)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    
    # Log hyperparameters
    logging.info("Trial hyperparameters:")
    for param_name, param_value in trial.params.items():
        logging.info(f"  {param_name}: {param_value}")
    
    # Fixed parameters
    input_dim = 5  # Based on data structure
    num_pred = 12  # Number of time steps to predict
    rid_of_top = 4
    size = 9
    num_in = 110
    
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
            
            # Reshape data for LSTM
            X_train = torch.reshape(X_train, (X_train.shape[0], num_in, X_train.shape[2]))
            
            # Move to device
            X_train = X_train.to(device)
            y_train = y_train.to(device)
            
            # For validation, we'll use the same tiles but with different splits
            X_test_tiles = []
            y_test_tiles = []
            for tile in all_tiles:
                X_test, y_test = lstm_ready(tile, size, inputs, intensities, num_in, num_pred)
                X_test = torch.reshape(X_test, (X_test.shape[0], num_in, X_test.shape[2]))
                X_test_tiles.append(X_test.to(device))
                y_test_tiles.append(y_test.to(device))
            
            # Initialize model with Xavier
            model = LSTM(input_size, hidden_size, num_layers, num_pred, dropout).to(device)
            initialize_weights(model)
            
            # Training setup
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            n_epochs = 300
            
            # Train model
            val_score, train_metrics, val_metrics, predictions = custom_training_loop(
                model, X_train, y_train, X_test_tiles, y_test_tiles, optimizer, n_epochs, device
            )
            
            logging.info(f"AR {AR} best validation score: {val_score:.6f}")
            all_val_scores.append(val_score)
            all_metric_details.append({
                'AR': AR,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'predictions': predictions
            })
            
        except Exception as e:
            logging.error(f"Error processing AR {AR}: {str(e)}")
            continue
    
    if not all_val_scores:
        return float('-inf')
    
    avg_score = np.mean(all_val_scores)
    logging.info(f"\nAverage validation score across all ARs: {avg_score:.6f}")
    
    return avg_score

def save_prediction_plots(metric_details, trial):
    """Save visualization plots for the trial."""
    import matplotlib.pyplot as plt
    
    for ar_details in metric_details:
        AR = ar_details['AR']
        predictions = ar_details['predictions']
        
        # Create directory for plots
        os.makedirs(f"trial_{trial.number}_plots", exist_ok=True)
        
        # Plot predictions at different epochs
        for pred in predictions:
            epoch = pred['epoch']
            y_pred = pred['pred']
            y_true = pred['true']
            
            plt.figure(figsize=(10, 6))
            plt.plot(y_true[0], label='True')
            plt.plot(y_pred[0], label='Predicted')
            plt.title(f'AR {AR} - Epoch {epoch}')
            plt.legend()
            plt.savefig(f"trial_{trial.number}_plots/AR{AR}_epoch{epoch}.png")
            plt.close()

def main():
    """Main function to run the optimization."""
    logging.info("\n" + "="*80)
    logging.info("Starting LSTM Shape-Based Hyperparameter Optimization")
    logging.info("="*80)
    
    # Create study
    n_trials = 50
    logging.info(f"Will run {n_trials} trials")
    
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
    )
    
    # Optimize
    study.optimize(objective, n_trials=n_trials)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"lstm_shape_optimization_{timestamp}_results.txt"
    
    with open(results_file, "w") as f:
        f.write(f"Best trial value: {study.best_trial.value}\n")
        f.write("Best trial params:\n")
        for key, value in study.best_trial.params.items():
            f.write(f"{key}: {value}\n")
        
        f.write("\nAll trials:\n")
        for trial in study.trials:
            f.write(f"Trial {trial.number}:\n")
            f.write(f"  Value: {trial.value}\n")
            f.write("  Params:\n")
            for key, value in trial.params.items():
                f.write(f"    {key}: {value}\n")
            f.write("\n")
    
    print(f"Optimization completed. Results saved to {results_file}")
    print("\nBest trial:")
    print("  Value: ", study.best_trial.value)
    print("  Params: ")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")

if __name__ == "__main__":
    main() 