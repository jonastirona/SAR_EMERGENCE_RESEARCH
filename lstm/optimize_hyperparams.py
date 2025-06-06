import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import optuna
from datetime import datetime
import os
import sys
from functions import split_image, get_piece_means, dtws, lstm_ready, LSTM, split_sequences, min_max_scaling

def load_and_preprocess_ar(AR, rid_of_top, size):
    # Load data
    power_maps = np.load('../data/AR{}/mean_pmdop{}_flat.npz'.format(AR,AR), allow_pickle=True)
    mag_flux = np.load('../data/AR{}/mean_mag{}_flat.npz'.format(AR,AR), allow_pickle=True)
    intensities = np.load('../data/AR{}/mean_int{}_flat.npz'.format(AR,AR), allow_pickle=True)
    
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
    # Define hyperparameters to optimize
    hidden_size = trial.suggest_int('hidden_size', 64, 512)
    num_layers = trial.suggest_int('num_layers', 1, 8)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    
    # Fixed parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = 5  # Based on your data structure
    num_pred = 12  # Number of time steps to predict
    rid_of_top = 4
    size = 9
    num_in = 110
    tiles = size**2 - 2*size*rid_of_top
    
    # Use multiple ARs for better generalization
    optimization_ARs = [11130, 11149, 11158, 11162, 11199, 11327, 11344, 11387, 11393, 11416]
    all_val_losses = []
    
    for AR in optimization_ARs:
        try:
            print(f"\nProcessing AR {AR}")
            # Load and preprocess data for this AR
            inputs, intensities = load_and_preprocess_ar(AR, rid_of_top, size)
            input_size = inputs.shape[1]
            
            # Create model
            model = LSTM(input_size, hidden_size, num_layers, num_pred, dropout=dropout).to(device)
            
            # Training setup
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = StepLR(optimizer, step_size=50, gamma=0.9)
            
            # Train on all available tiles
            ar_val_losses = []
            for tile in range(tiles):
                print(f"Training on tile {tile}")
                # Split data into train/val sets
                X_data, y_data = lstm_ready(tile, size, inputs, intensities, num_in, num_pred)
                train_size = int(0.8 * len(X_data))
                
                X_train = X_data[:train_size]
                y_train = y_data[:train_size]
                X_val = X_data[train_size:]
                y_val = y_data[train_size:]
                
                # Reshape data
                X_train = torch.reshape(X_train, (X_train.shape[0], num_in, X_train.shape[2]))
                X_val = torch.reshape(X_val, (X_val.shape[0], num_in, X_val.shape[2]))
                
                # Move data to device
                X_train = X_train.to(device)
                y_train = y_train.to(device)
                X_val = X_val.to(device)
                y_val = y_val.to(device)
                
                # Training loop
                n_epochs = 100  # Reduced epochs for optimization
                best_val_loss = float('inf')
                
                for epoch in range(n_epochs):
                    model.train()
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = model(X_train)
                    loss = criterion(outputs, y_train)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    # Validation
                    model.eval()
                    with torch.no_grad():
                        val_outputs = model(X_val)
                        val_loss = criterion(val_outputs, y_val)
                        
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss.item()
                    
                    scheduler.step()
                    
                    # Optuna pruning
                    trial.report(val_loss.item(), epoch)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                
                ar_val_losses.append(best_val_loss)
                print(f"Tile {tile} best validation loss: {best_val_loss:.6f}")
            
            # Average validation loss across all tiles for this AR
            ar_avg_loss = np.mean(ar_val_losses)
            all_val_losses.append(ar_avg_loss)
            print(f"AR {AR} average validation loss: {ar_avg_loss:.6f}")
            
        except Exception as e:
            print(f"Error processing AR {AR}: {str(e)}")
            continue
    
    # Return average validation loss across all successfully processed ARs
    if not all_val_losses:
        return float('inf')
    
    final_loss = np.mean(all_val_losses)
    print(f"\nTrial average validation loss across all ARs: {final_loss:.6f}")
    return final_loss

def main():
    # Create study
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_name = f"lstm_optimization_{timestamp}"
    
    # Create results directory if it doesn't exist
    os.makedirs("optuna_results", exist_ok=True)
    
    # Create study
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        study_name=study_name
    )
    
    # Optimize
    n_trials = 50
    study.optimize(objective, n_trials=n_trials)
    
    # Save results
    results_file = f"optuna_results/{study_name}_results.txt"
    with open(results_file, "w") as f:
        f.write(f"Best trial value: {study.best_trial.value}\n")
        f.write("Best trial params:\n")
        for key, value in study.best_trial.params.items():
            f.write(f"{key}: {value}\n")
        
        f.write("\nAll trials:\n")
        for i, trial in enumerate(study.trials):
            f.write(f"Trial {i}:\n")
            f.write(f"  Value: {trial.value}\n")
            f.write("  Params:\n")
            for key, value in trial.params.items():
                f.write(f"    {key}: {value}\n")
            f.write("\n")
    
    # Create visualization plots
    import plotly
    fig1 = optuna.visualization.plot_optimization_history(study)
    fig2 = optuna.visualization.plot_parallel_coordinate(study)
    fig3 = optuna.visualization.plot_param_importances(study)
    
    # Save plots
    fig1.write_html(f"optuna_results/{study_name}_optimization_history.html")
    fig2.write_html(f"optuna_results/{study_name}_parallel_coordinate.html")
    fig3.write_html(f"optuna_results/{study_name}_param_importances.html")
    
    print(f"Optimization completed. Results saved to {results_file}")
    print("\nBest trial:")
    print("  Value: ", study.best_trial.value)
    print("  Params: ")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")

if __name__ == "__main__":
    main() 