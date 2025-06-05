import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import numpy as np
import sys
from PIL import Image
from sklearn.model_selection import train_test_split
from functions import split_image, get_piece_means, dtws, lstm_ready, training_loop, training_loop_w_stats, LSTM, split_sequences, min_max_scaling, calculate_metrics
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ExponentialLR, ReduceLROnPlateau
from torch.utils.data import Dataset, TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import os
from matplotlib.backends.backend_pdf import PdfPages

start_time = time.time()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Define the device (either 'cuda' for GPU or 'cpu' for CPU)
print('Runs on: {}'.format(device))
print("Using", torch.cuda.device_count(), "GPUs!")

# Check if the correct number of arguments is provided
if len(sys.argv) != 9:
    print("Usage: script.py num_ARs n_epochs learning_rate rid_of_top hidden_size dropout")
    sys.exit(1)
try: # Extract arguments and convert them to the appropriate types
    num_pred = int(sys.argv[1]); print("Time Windows:", num_pred)
    rid_of_top = int(sys.argv[2]); print("Rid of Top:", rid_of_top)
    num_in = int(sys.argv[3]); print("Number of Inputs:", num_in)
    num_layers = int(sys.argv[4]); print("Number of Layers:", num_layers)
    hidden_size = int(sys.argv[5]); print("Hidden Size:", hidden_size)
    n_epochs = int(sys.argv[6]); print("Number of Epochs:", n_epochs)
    learning_rate = float(sys.argv[7]); print("Learning Rate:", learning_rate)
    dropout = float(sys.argv[8]); print("Dropout:", dropout)
except ValueError as e:
    print("Error: Please ensure that all arguments are numbers.")
    sys.exit(1)

# Now you can use these variables in your script
ARs = [11130,11149,11158,11162,11199,11327,11344,11387,11393,11416,11422,11455,11619,11640,11660,11678,11682,11765,11768,11776,11916,11928,12036,12051,12085,12089,12144,12175,12203,12257,12331,12494,12659,12778,12864,12877,12900,12929,13004,13085,13098]
flatten = True
size = 9
tiles = size**2 - 2*size*rid_of_top
test_AR = 13179 # and the secondary will be 13165 and if I fix it, third: 13183
ARs_ = ARs + [test_AR]

#Preprocessing
print('Load data and split in tiles for {} ARs'.format(len(ARs)))
all_inputs = []
all_intensities = []
for AR in ARs_:
    power_maps = np.load('/mmfs1/project/mx6/jst26/SAR_EMERGENCE_RESEARCH/data/AR{}/mean_pmdop{}_flat.npz'.format(AR,AR),allow_pickle=True) 
    mag_flux = np.load('/mmfs1/project/mx6/jst26/SAR_EMERGENCE_RESEARCH/data/AR{}/mean_mag{}_flat.npz'.format(AR,AR),allow_pickle=True)
    intensities = np.load('/mmfs1/project/mx6/jst26/SAR_EMERGENCE_RESEARCH/data/AR{}/mean_int{}_flat.npz'.format(AR,AR),allow_pickle=True)
    power_maps23 = power_maps['arr_0']
    power_maps34 = power_maps['arr_1']
    power_maps45 = power_maps['arr_2']
    power_maps56 = power_maps['arr_3']
    mag_flux = mag_flux['arr_0']
    intensities = intensities['arr_0']
    # Trim array to get rid of top and bottom 0 tiles
    power_maps23 = power_maps23[rid_of_top*size:-rid_of_top*size, :] 
    power_maps34 = power_maps34[rid_of_top*size:-rid_of_top*size, :]
    power_maps45 = power_maps45[rid_of_top*size:-rid_of_top*size, :]
    power_maps56 = power_maps56[rid_of_top*size:-rid_of_top*size, :]
    mag_flux = mag_flux[rid_of_top*size:-rid_of_top*size, :] ; mag_flux[np.isnan(mag_flux)] = 0
    intensities = intensities[rid_of_top*size:-rid_of_top*size, :] ; intensities[np.isnan(intensities)] = 0
    # stack inputs and normalize
    stacked_maps = np.stack([power_maps23, power_maps34, power_maps45, power_maps56], axis=1); stacked_maps[np.isnan(stacked_maps)] = 0
    min_p = np.min(stacked_maps); max_p = np.max(stacked_maps)
    min_m = np.min(mag_flux); max_m = np.max(mag_flux)
    min_i = np.min(intensities); max_i = np.max(intensities)
    stacked_maps = min_max_scaling(stacked_maps, min_p, max_p)
    mag_flux = min_max_scaling(mag_flux, min_m, max_m)
    intensities = min_max_scaling(intensities, min_i, max_i)
    # Reshape mag_flux to have an extra dimension and then put it with pmaps
    mag_flux_reshaped = np.expand_dims(mag_flux, axis=1)
    pm_and_flux = np.concatenate([stacked_maps, mag_flux_reshaped], axis=1)
    # append all ARs
    all_inputs.append(pm_and_flux)
    all_intensities.append(intensities)
all_inputs = np.stack(all_inputs, axis=-1)
all_intensities = np.stack(all_intensities, axis=-1)
input_size = np.shape(all_inputs)[1]

# Start Training
lstm = LSTM(input_size, hidden_size, num_layers, num_pred, dropout=dropout).to(device)
#if torch.cuda.device_count() > 1: lstm = torch.nn.DataParallel(lstm)
loss_fn = torch.nn.MSELoss()  #torch.nn.L1Loss() #   # mean-squared error for regression
#optimiser = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

# Define paths for results
base_path = './results'  # Creates results in lstm/results/
model_name_str = 't{}_r{}_i{}_n{}_h{}_e{}_l{}'.format(num_pred,rid_of_top,num_in,num_layers,hidden_size,n_epochs,learning_rate)
result_file_path = os.path.join(base_path, f"{model_name_str}_training_stats.txt")
pdf_path = os.path.join(base_path, f"{model_name_str}_loss_curves.pdf")
model_path = os.path.join(base_path, f"{model_name_str}.pth")

os.makedirs(base_path, exist_ok=True)

# Dictionary to store all results
all_training_results = {}

# Open the file for writing training stats
with open(result_file_path, "w") as file:
    # Iterate over ARs and tiles, writing results to the same file
    for AR_ in range(len(ARs)):
        power_maps = all_inputs[:,:,:,AR_] #change to inputs, its not only power maps
        intensities = all_intensities[:,:,AR_]
        for tile in range(tiles):
            optimiser = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
            print('AR{} - Tile: {}'.format(ARs[AR_],tile))
            X_train, y_train = lstm_ready(tile,size,power_maps,intensities,num_in,num_pred)
            X_test, y_test = lstm_ready(int(tiles/2),size,power_maps,intensities,num_in,num_pred)
            # reshaping to rows, timestamps, features
            X_train_final = torch.reshape(X_train,(X_train.shape[0], num_in, X_train.shape[2]))
            X_test_final = torch.reshape(X_test,(X_test.shape[0], num_in, X_test.shape[2])) 
            # Move data to GPU
            X_train_final = X_train_final.to(device)
            y_train = y_train.to(device)
            X_test_final = X_test_final.to(device)
            y_test = y_test.to(device)
            results = training_loop_w_stats(n_epochs=n_epochs,lstm=lstm,optimiser=optimiser,loss_fn=loss_fn,
                        X_train=X_train_final,
                        y_train=y_train,
                        X_test=X_test_final,
                        y_test=y_test)
            
            # Store results for plotting
            all_training_results[f'AR{ARs[AR_]}_Tile{tile}'] = results
            
            # Write AR and tile header
            file.write(f"\nAR {ARs[AR_]} - Tile {tile}\n")
            file.write("Epoch, Train Loss, Test Loss, Learning Rate\n")
            for epoch, train_loss, test_loss, lr in results:
                file.write(f"{epoch}, {train_loss:.5f}, {test_loss:.5f}, {lr:.5f}\n")

# Create PDF with loss curves
with PdfPages(pdf_path) as pdf:
    # Create summary plots
    fig = plt.figure(figsize=(15, 10))
    gs = plt.GridSpec(2, 2, figure=fig)
    
    # Plot 1: Average losses across all tiles
    ax1 = fig.add_subplot(gs[0, :])
    avg_train_losses = np.mean([np.array([r[1] for r in results]) for results in all_training_results.values()], axis=0)
    avg_test_losses = np.mean([np.array([r[2] for r in results]) for results in all_training_results.values()], axis=0)
    epochs = range(n_epochs)
    
    ax1.plot(epochs, avg_train_losses, label='Avg Training Loss')
    ax1.plot(epochs, avg_test_losses, label='Avg Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Average Loss')
    ax1.set_title('Average Training and Test Loss Across All Tiles')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Average learning rate
    ax2 = fig.add_subplot(gs[1, 0])
    avg_lr = np.mean([np.array([r[3] for r in results]) for results in all_training_results.values()], axis=0)
    ax2.plot(epochs, avg_lr, label='Avg Learning Rate', color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Average Learning Rate Schedule')
    ax2.legend()
    ax2.grid(True)
    
    # Plot 3: Evaluation Metrics Distribution
    ax3 = fig.add_subplot(gs[1, 1])
    all_metrics = []
    
    # Calculate metrics for each AR and tile
    for key, results in all_training_results.items():
        # Get the final predictions for this AR/tile
        final_predictions = lstm(X_test_final).detach().cpu().numpy()
        true_values = y_test.cpu().numpy()
        
        # Calculate metrics
        metrics = calculate_metrics(true_values, final_predictions)
        all_metrics.append(metrics)
    
    # Convert to numpy array for easier manipulation
    all_metrics = np.array(all_metrics)
    
    # Create box plot of metrics
    metric_names = ['MAE', 'MSE', 'RMSE', 'RMSLE', 'R²']
    bp = ax3.boxplot(all_metrics, labels=metric_names)
    ax3.set_title('Distribution of Evaluation Metrics')
    ax3.grid(True)
    
    # Add mean values as text
    mean_metrics = np.mean(all_metrics, axis=0)
    textstr = '\n'.join([f'{name}: {val:.4f}' for name, val in zip(metric_names, mean_metrics)])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax3.text(0.05, 0.95, textstr, transform=ax3.transAxes, fontsize=8,
             verticalalignment='top', bbox=props)
    
    plt.suptitle(f'Training Summary - LSTM\nParameters: TW={num_pred}, RoT={rid_of_top}, In={num_in}, Layers={num_layers}, Hidden={hidden_size}', 
                 fontsize=12)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

# Save the model weights
torch.save(lstm.state_dict(), model_path)
end_time = time.time()
print("Elapsed time: {} minutes".format((end_time - start_time)/60))
print(f"Results saved at: {result_file_path}")
print(f"Loss curves saved at: {pdf_path}")
print(f"Model saved at: {model_path}")

# Print final average metrics
print("\nFinal Average Metrics:")
for name, value in zip(metric_names, mean_metrics):
    print(f"{name}: {value:.4f}")