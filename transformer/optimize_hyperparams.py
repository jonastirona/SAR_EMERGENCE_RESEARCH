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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def load_and_preprocess_ar(AR, rid_of_top, size):
    # Load data
    power_maps = np.load(f'/mmfs1/project/mx6/jst26/data/AR{AR}/mean_pmdop{AR}_flat.npz', allow_pickle=True)
    mag_flux = np.load(f'/mmfs1/project/mx6/jst26/data/AR{AR}/mean_mag{AR}_flat.npz', allow_pickle=True)
    intensities = np.load(f'/mmfs1/project/mx6/jst26/data/AR{AR}/mean_int{AR}_flat.npz', allow_pickle=True)

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
    n_epochs = 300  # Increased epochs for better convergence
    size = 9
    
    # Hyperparameters to optimize
    # First select number of heads (must be a power of 2)
    num_heads = trial.suggest_int('num_heads', 2, 16, step=2)
    
    # Then select hidden_size that's divisible by num_heads
    hidden_size_choices = list(range(num_heads, 513, num_heads))  # Increased max hidden size
    hidden_size = trial.suggest_int('hidden_size', hidden_size_choices[0], hidden_size_choices[-1], step=num_heads)
    
    # Other hyperparameters
    num_layers = trial.suggest_int('num_layers', 1, 8)  # Increased max layers
    ff_ratio = trial.suggest_float('ff_ratio', 2.0, 8.0)  # Increased FF ratio range
    ff_dim = int(hidden_size * ff_ratio)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)  # Extended LR range
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    warmup_ratio = trial.suggest_float('warmup_ratio', 0.0, 0.2)  # 0-20% of total epochs for warmup

    # Log hyperparameters
    logging.info("Trial hyperparameters:")
    logging.info(f"  hidden_size: {hidden_size}")
    logging.info(f"  num_layers: {num_layers}")
    logging.info(f"  num_heads: {num_heads}")
    logging.info(f"  ff_ratio: {ff_ratio:.2f} (ff_dim: {ff_dim})")
    logging.info(f"  learning_rate: {learning_rate:.6f}")
    logging.info(f"  dropout: {dropout:.2f}")
    logging.info(f"  warmup_ratio: {warmup_ratio:.2f}")
    logging.info(f"  hidden_size divisible by num_heads: {hidden_size % num_heads == 0}")

    # Use multiple ARs for better generalization
    optimization_ARs = [11130, 11149, 11158, 11162, 11199, 11327, 11344, 11387, 11393, 11416]
    all_val_losses = []
    
    for AR in optimization_ARs:
        try:
            logging.info(f"\nProcessing AR {AR}")
            
            # Load and preprocess data
            inputs, intensities = load_and_preprocess_ar(AR, rid_of_top, size)
            input_size = inputs.shape[1]

    # Prepare data for model
            middle_tile = (size**2 - 2*size*rid_of_top) // 2  # Use middle tile for training
            X_train, y_train = lstm_ready(middle_tile, size, inputs, intensities, num_in, num_pred)
            X_test, y_test = lstm_ready(0, size, inputs, intensities, num_in, num_pred)  # Use first tile for validation

    # Move data to device
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    # Initialize model
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

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training
        results = training_loop_w_stats(
            n_epochs=n_epochs,
            lstm=model,
            optimiser=optimizer,
            loss_fn=criterion,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            use_scheduler=True
        )
        
        # Get the best validation loss
        _, _, test_losses, _ = zip(*results)
        best_val_loss = min(test_losses)
            all_val_losses.append(best_val_loss)
        
            logging.info(f"AR {AR} best validation loss: {best_val_loss:.6f}")
    
    except Exception as e:
            logging.error(f"Error processing AR {AR}: {str(e)}")
            continue
    
    # Return average validation loss across all successfully processed ARs
    if not all_val_losses:
        return float('inf')
    
    avg_loss = np.mean(all_val_losses)
    logging.info(f"\nAverage validation loss across all ARs: {avg_loss:.6f}")
    return avg_loss

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