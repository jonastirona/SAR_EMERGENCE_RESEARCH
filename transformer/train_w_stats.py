import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import numpy as np
import sys
from PIL import Image
from sklearn.model_selection import train_test_split
from functions import split_image, get_piece_means, dtws, lstm_ready, training_loop, training_loop_w_stats, LSTM, split_sequences, min_max_scaling, calculate_metrics, calculate_extended_metrics
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
from models.st_transformer import SpatioTemporalTransformer
from matplotlib.backends.backend_pdf import PdfPages
import logging
import traceback

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def validate_parameters(num_pred, rid_of_top, num_in, num_layers, hidden_size, n_epochs, 
                       learning_rate, model_name, embed_dim, num_heads, ff_dim, dropout):
    """Validate that all parameters are within expected ranges."""
    try:
        assert 1 <= num_pred <= 24, f"num_pred should be between 1 and 24, got {num_pred}"
        assert 0 <= rid_of_top <= 4, f"rid_of_top should be between 0 and 4, got {rid_of_top}"
        assert 10 <= num_in <= 200, f"num_in should be between 10 and 200, got {num_in}"
        assert 1 <= num_layers <= 6, f"num_layers should be between 1 and 6, got {num_layers}"
        assert 32 <= hidden_size <= 512, f"hidden_size should be between 32 and 512, got {hidden_size}"
        assert 1 <= n_epochs <= 1000, f"n_epochs should be between 1 and 1000, got {n_epochs}"
        assert 0.0001 <= learning_rate <= 0.1, f"learning_rate should be between 0.0001 and 0.1, got {learning_rate}"
        assert model_name in ['LSTM', 'st_transformer'], f"model_name should be 'LSTM' or 'st_transformer', got {model_name}"
        assert 32 <= embed_dim <= 512, f"embed_dim should be between 32 and 512, got {embed_dim}"
        assert 1 <= num_heads <= 16, f"num_heads should be between 1 and 16, got {num_heads}"
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        assert 64 <= ff_dim <= 1024, f"ff_dim should be between 64 and 1024, got {ff_dim}"
        assert 0 <= dropout <= 0.5, f"dropout should be between 0 and 0.5, got {dropout}"
        return True
    except AssertionError as e:
        logger.error(f"Parameter validation failed: {str(e)}")
        return False

def main():
    try:
        start_time = time.time()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info('Runs on: {}'.format(device))
        logger.info("Using {} GPUs!".format(torch.cuda.device_count()))

        # Check if correct number of arguments is provided
        if len(sys.argv) != 13:
            logger.error(f"Expected 13 arguments, got {len(sys.argv)}")
            logger.error("Usage: python train_w_stats.py num_pred rid_of_top num_in num_layers hidden_size n_epochs learning_rate embed_dim num_heads ff_dim dropout save_dir")
            sys.exit(1)

        # Parse command line arguments
        num_pred = int(sys.argv[1])
        rid_of_top = 4
        num_in = int(sys.argv[3])
        num_layers = int(sys.argv[4])
        hidden_size = int(sys.argv[5])
        n_epochs = int(sys.argv[6])
        learning_rate = float(sys.argv[7])
        embed_dim = int(sys.argv[8])
        num_heads = int(sys.argv[9])
        ff_dim = int(sys.argv[10])
        dropout = float(sys.argv[11])
        save_dir = sys.argv[12]

        # Set default model name
        model_name = 'st_transformer'

        # Validate parameters
        if not validate_parameters(num_pred, rid_of_top, num_in, num_layers, hidden_size, n_epochs,
                                 learning_rate, model_name, embed_dim, num_heads, ff_dim, dropout):
            sys.exit(1)

        logger.info("Starting training with parameters:")
        logger.info(f"num_pred: {num_pred}")
        logger.info(f"rid_of_top: {rid_of_top}")
        logger.info(f"num_in: {num_in}")
        logger.info(f"num_layers: {num_layers}")
        logger.info(f"hidden_size: {hidden_size}")
        logger.info(f"n_epochs: {n_epochs}")
        logger.info(f"learning_rate: {learning_rate}")
        logger.info(f"model_name: {model_name}")
        logger.info(f"embed_dim: {embed_dim}")
        logger.info(f"num_heads: {num_heads}")
        logger.info(f"ff_dim: {ff_dim}")
        logger.info(f"dropout: {dropout}")
        logger.info(f"save_dir: {save_dir}")

        # Define test AR and size
        test_AR = 13179
        size = 9  # Define size here
        remaining_rows = size - 2*rid_of_top  # Now rid_of_top is defined
        tiles = remaining_rows * size
        logger.info(f"Using test AR: {test_AR}")
        logger.info(f"Size: {size}")
        logger.info(f"Remaining rows: {remaining_rows}")
        logger.info(f"Total tiles: {tiles}")

        # Load data
        logger.info("Loading data...")
        power_maps = np.load('/mmfs1/project/mx6/jst26/SAR_EMERGENCE_RESEARCH/data/AR{}/mean_pmdop{}_flat.npz'.format(test_AR, test_AR), allow_pickle=True)['arr_0']
        mag_flux = np.load('/mmfs1/project/mx6/jst26/SAR_EMERGENCE_RESEARCH/data/AR{}/mean_mag{}_flat.npz'.format(test_AR, test_AR), allow_pickle=True)['arr_0']
        intensities = np.load('/mmfs1/project/mx6/jst26/SAR_EMERGENCE_RESEARCH/data/AR{}/mean_int{}_flat.npz'.format(test_AR, test_AR), allow_pickle=True)['arr_0']
        logger.info(f"Power maps shape: {power_maps.shape}")
        logger.info(f"Intensities shape: {intensities.shape}")

        # Prepare data for a specific tile (e.g., tile 0)
        logger.info("Preparing data for training...")
        X, y = lstm_ready(0, size, power_maps, intensities, num_in, num_pred)  # Pass size here
        logger.info(f"X shape before reshape: {X.shape}")
        logger.info(f"y shape: {y.shape}")

        # Reshape X to match transformer input format (batch, seq_len, features)
        X = X.unsqueeze(-1)  # Add feature dimension
        logger.info(f"X shape after reshape: {X.shape}")

        # Split data into train and test sets
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        logger.info(f"Train set size: {len(X_train)}")
        logger.info(f"Test set size: {len(X_test)}")

        # Initialize model
        logger.info("Initializing model...")
        if model_name == 'LSTM':
            model = LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers, output_length=num_pred)
        elif model_name == 'st_transformer':
            model = SpatioTemporalTransformer(
                input_dim=1,  # Single feature dimension
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                ff_dim=ff_dim,
                dropout=dropout,
                output_dim=num_pred
            )
        else:
            logger.error(f"Unknown model name: {model_name}")
            sys.exit(1)

        logger.info(f"Model architecture: {model}")
        logger.info(f"Input shape: {X.shape}")
        logger.info(f"Output shape: {y.shape}")

        # Initialize optimizer and loss function
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()

        # Training loop
        logger.info("Starting training loop...")
        results = training_loop_w_stats(n_epochs, model, optimizer, loss_fn, X_train, y_train, X_test, y_test)
        logger.info("Training completed")

        # Calculate metrics
        logger.info("Calculating metrics...")
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test)
            metrics = calculate_extended_metrics(model, y_test.numpy(), y_pred.numpy())
            logger.info(f"Final metrics: {metrics}")

        # Save results
        logger.info("Saving results...")
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, 'results.npy'), results)
        np.save(os.path.join(save_dir, 'metrics.npy'), metrics)
        logger.info("Results saved successfully")

        end_time = time.time()
        logger.info("Elapsed time: {} minutes".format((end_time - start_time)/60))

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.error("Traceback:")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()

start_time = time.time()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Define the device (either 'cuda' for GPU or 'cpu' for CPU)
print('Runs on: {}'.format(device))
print("Using", torch.cuda.device_count(), "GPUs!")

# Check if correct number of arguments is provided
if len(sys.argv) != 13:
    logger.error(f"Expected 13 arguments, got {len(sys.argv)}")
    logger.error("Usage: python train_w_stats.py num_pred rid_of_top num_in num_layers hidden_size n_epochs learning_rate embed_dim num_heads ff_dim dropout save_dir")
    sys.exit(1)

# Parse command line arguments
num_pred = int(sys.argv[1])
rid_of_top = 4
num_in = int(sys.argv[3])
num_layers = int(sys.argv[4])
hidden_size = int(sys.argv[5])
n_epochs = int(sys.argv[6])
learning_rate = float(sys.argv[7])
embed_dim = int(sys.argv[8])
num_heads = int(sys.argv[9])
ff_dim = int(sys.argv[10])
dropout = float(sys.argv[11])
save_dir = sys.argv[12]

# Set default model name
model_name = 'st_transformer'

# Now you can use these variables in your script
ARs = [11130,11149,11158,11162,11199,11327,11344,11387,11393,11416,11422,11455,11619,11640,11660,11678,11682,11765,11768,11776,11916,11928,12036,12051,12085,12089,12144,12175,12203,12257,12331,12494,12659,12778,12864,12877,12900,12929,13004,13085,13098]
flatten = True
size = 9
remaining_rows = size - 2*rid_of_top  # Calculate remaining rows after removing top and bottom
tiles = remaining_rows * size  # Total number of tiles is remaining rows times width
test_AR = 13179 # and the secondary will be 13165 and if I fix it, third: 13183
ARs_ = ARs + [test_AR]

# Function to convert linear tile index to row, col format
def get_tile_position(tile_idx, size):
    row = tile_idx // size
    col = tile_idx % size
    return row, col

# Function to get valid tile index
def get_valid_tile_idx(row, col, size):
    return row * size + col

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
if model_name == 'lstm':
    model = LSTM(input_size, hidden_size, num_layers, num_pred).to(device)
elif model_name == 'st_transformer':
    # Initialize transformer with optimized parameters
    model = SpatioTemporalTransformer(
        input_dim=input_size,
        seq_len=num_in,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_layers=num_layers,
        output_dim=num_pred,
        dropout=dropout
    ).to(device)

#if torch.cuda.device_count() > 1: lstm = torch.nn.DataParallel(lstm)
loss_fn = torch.nn.MSELoss()  #torch.nn.L1Loss() #   # mean-squared error for regression
#optimiser = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

# Define paths for results
base_path = save_dir
base_path = os.path.join(base_path, model_name)
model_name_str = 't{}_r{}_i{}_n{}_h{}_e{}_l{}_ed{}_nh{}_ff{}_d{}'.format(
    num_pred, rid_of_top, num_in, num_layers, hidden_size, n_epochs, learning_rate,
    embed_dim, num_heads, ff_dim, dropout
)
result_file_path = os.path.join(base_path, f"{model_name_str}_training_stats.txt")
pdf_path = os.path.join(base_path, f"{model_name_str}_loss_curves.pdf")
model_path = os.path.join(base_path, f"{model_name_str}.pth")

os.makedirs(base_path, exist_ok=True)

# Dictionary to store all results
all_training_results = {}

# Open the file for writing training stats
with open(result_file_path, "w") as file:
    # Iterate over ARs, writing results to the same file
    for AR_ in range(len(ARs)):
        power_maps = all_inputs[:,:,:,AR_] 
        intensities = all_intensities[:,:,AR_]
        
        # Prepare data from all tiles
        X_trains = []
        y_trains = []
        
        # Generate valid tile indices
        valid_tiles = []
        for tile in range(tiles):
            row = tile // size
            col = tile % size
            valid_tiles.append(col)  # We only need the column index since rows are already trimmed
        
        # Collect data from all tiles
        for tile in valid_tiles:
            X_tile, y_tile = lstm_ready(tile, size, power_maps, intensities, num_in, num_pred)
            X_trains.append(X_tile)
            y_trains.append(y_tile)
        
        # Concatenate all tile data
        X_train = torch.cat(X_trains, dim=0)
        y_train = torch.cat(y_trains, dim=0)
        
        # Reshape data for transformer
        X_train = torch.reshape(X_train, (X_train.shape[0], num_in, X_train.shape[2]))
        
        # Move data to GPU
        X_train = X_train.to(device)
        y_train = y_train.to(device)
        
        # For validation, use all tiles with different splits
        X_test_tiles = []
        y_test_tiles = []
        for tile in valid_tiles:
            X_test, y_test = lstm_ready(tile, size, power_maps, intensities, num_in, num_pred)
            X_test = torch.reshape(X_test, (X_test.shape[0], num_in, X_test.shape[2]))
            X_test_tiles.append(X_test.to(device))
            y_test_tiles.append(y_test.to(device))
        
        # Initialize optimizer for this AR
        optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
        print(f'Training on AR{ARs[AR_]} - All Tiles')
        
        # Train on all tiles together
        results = training_loop_w_stats(
            n_epochs=n_epochs,
            lstm=model,
            optimiser=optimiser,
            loss_fn=loss_fn,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test_tiles[0],  # Use first tile for main validation
            y_test=y_test_tiles[0],
            use_scheduler=True
        )
        
        # Store results for plotting
        all_training_results[f'AR{ARs[AR_]}'] = results
        
        # Write AR header
        file.write(f"\nAR {ARs[AR_]} - All Tiles\n")
        file.write("Epoch, Train Loss, Test Loss, Learning Rate\n")
        for epoch, train_loss, test_loss, lr in results:
            file.write(f"{epoch}, {train_loss:.5f}, {test_loss:.5f}, {lr:.5f}\n")
            
        # Evaluate on each tile separately for detailed metrics
        file.write("\nPer-tile validation metrics after training:\n")
        for tile in valid_tiles:
            with torch.no_grad():
                test_pred = model(X_test_tiles[tile])
                test_loss = loss_fn(test_pred, y_test_tiles[tile])
                file.write(f"Tile {tile} - Test Loss: {test_loss:.5f}\n")

# Create PDF with loss curves
with PdfPages(pdf_path) as pdf:
    # Create summary plots
    fig = plt.figure(figsize=(15, 12))  # Increased height for new metrics
    gs = plt.GridSpec(3, 2, figure=fig)  # Changed to 3 rows
    
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
    training_time = (time.time() - start_time) / 60  # Convert to minutes
    for key, results in all_training_results.items():
        # Extract AR number from the key (format is 'AR{number}')
        ar_number = int(key.replace('AR', ''))
        # Use the first tile for metrics calculation
        final_predictions = model(X_test_tiles[0]).detach().cpu().numpy()
        true_values = y_test_tiles[0].cpu().numpy()
        
        # Calculate extended metrics
        metrics = calculate_extended_metrics(model, true_values, final_predictions, training_time)
        all_metrics.append(metrics)
    
    # Convert metrics to arrays for plotting
    metric_names = ['MAE', 'RMSE', 'R2', 'RMSE@1', 'RMSE@5']
    metric_values = [[m[name] for m in all_metrics] for name in metric_names]
    
    # Create box plot of metrics
    bp = ax3.boxplot(metric_values, labels=metric_names)
    ax3.set_title('Distribution of Evaluation Metrics')
    ax3.grid(True)
    
    # Add mean values as text
    mean_metrics = {name: np.mean([m[name] for m in all_metrics]) for name in metric_names}
    param_count = all_metrics[0]['params']  # Same for all tiles
    
    # Plot 4: Extended Metrics Table
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    
    # Create table data
    table_data = [
        ['Metric', 'Value'],
        ['Parameters', f'{param_count/1e6:.1f}M'],
        ['Training Time', f'{training_time:.1f} min'],
        ['MAE', f'{mean_metrics["MAE"]:.3f}'],
        ['RMSE', f'{mean_metrics["RMSE"]:.3f}'],
        ['R²', f'{mean_metrics["R2"]:.3f}'],
        ['RMSE@1', f'{mean_metrics["RMSE@1"]:.3f}'],
        ['RMSE@5', f'{mean_metrics["RMSE@5"]:.3f}' if mean_metrics["RMSE@5"] is not None else 'N/A']
    ]
    
    table = ax4.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    plt.suptitle(f'Training Summary - Transformer\nParameters: TW={num_pred}, RoT={rid_of_top}, In={num_in}, Embed={hidden_size}, Layers={num_layers}', 
                 fontsize=12)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

# Save the model weights
torch.save(model.state_dict(), model_path)
end_time = time.time()
print("Elapsed time: {} minutes".format((end_time - start_time)/60))
print(f"Results saved at: {result_file_path}")
print(f"Loss curves saved at: {pdf_path}")
print(f"Model saved at: {model_path}")

# Print final metrics in table format
print("\nFinal Metrics:")
print("-" * 40)
print(f"{'Metric':<15} {'Value':<10}")
print("-" * 40)
for row in table_data[1:]:
    print(f"{row[0]:<15} {row[1]:<10}")
print("-" * 40)

