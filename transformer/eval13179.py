from functions import split_image, get_piece_means, dtws, lstm_ready, training_loop, LSTM, split_sequences, min_max_scaling, amplify_fluctuations, calculate_metrics, emergence_indication, smooth_with_numpy, recalibrate, find_closest_fits_frame_to_NOAA_record, add_grid_lines, highlight_tile, emergence_indication2
from torch.optim.lr_scheduler import StepLR, ExponentialLR, ReduceLROnPlateau
from torch.utils.data import Dataset, TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
from collections import OrderedDict
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import gridspec
from datetime import datetime
import torch.optim as optim
import torch.nn as nn
from PIL import Image
import numpy as np
import warnings
import torch
import sys
import glob
import re
import os

# Initialize and load network
warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Define the device (either 'cuda' for GPU or 'cpu' for CPU)
print('Runs on: {}'.format(device)," / Using", torch.cuda.device_count(), "GPUs!")
pth_file = glob.glob('./ApJ_paper_lstm/best/*.pth') # Assuming there's only one .pth file and its naming follows the specific pattern
filename = pth_file[0]
matches = re.findall(r't(\d+)_r(\d+)_i(\d+)_n(\d+)_h(\d+)_e(\d+)_l([0-9.]+)\.pth', filename) # Extract numbers from the filename
num_pred, rid_of_top, num_in, num_layers, hidden_size, n_epochs, learning_rate = (int(x) if i != 6 else float(x) for i, x in enumerate(matches[0])) # Unpack the matched values into variables
print(f"Extracted from filename: Time Windows: {num_pred}, Rid of Top: {rid_of_top}, Number of Inputs: {num_in}, Number of Layers: {num_layers}, Hidden Size: {hidden_size}, Number of Epochs: {n_epochs}, Learning Rate: {learning_rate}") # Print extracted values for confirmation

# Now you can use these variables in your script
rid_of_top = 1; num_in = 96 #REDEFINED BE CAREFUL / # Now you can use these variables in your script
ARs = [11130,11149,11158,11162,11199,11327,11344,11387,11393,11416,11422,11455,11619,11640,11660,11678,11682,11765,11768,11776,11916,11928,12036,12051,12085,12089,12144,12175,12203,12257,12331,12494,12659,12778,12864,12877,12900,12929,13004,13085,13098]
flatten = True
size = 9
tiles = size**2 - 2*size*rid_of_top
test_AR = 13179 # and the secondary will be 13165 and if I fix it, third: 13183
intensity_files = glob.glob(os.path.join("/home/jonas/Documents/AR_PREDICTION_MODELS/data/AR{}/cont_intensity/".format(test_AR), "*.fits")) # get a list of all files with a .fits extension in the directory
NOAA_first = datetime(2022, 12, 30, 0, 0, 0); NOAA_first_record = mdates.date2num(NOAA_first) # Convert to matplotlib date format
NOAA_second = datetime(2023, 1, 1, 0, 0, 0); NOAA_second_record = mdates.date2num(NOAA_second) # Convert to matplotlib date format
NOAA_first_int_map = find_closest_fits_frame_to_NOAA_record(intensity_files, NOAA_first)
NOAA_second_int_map = find_closest_fits_frame_to_NOAA_record(intensity_files, NOAA_second)
#NOAA_first_int_map = np.random.rand(512, 512)  # Random noise image
#NOAA_second_int_map = np.random.rand(512, 512)  # Another random noise image

#Preprocessing
power_maps = np.load('/home/jonas/Documents/AR_PREDICTION_MODELS/data/AR{}/mean_tiles9/mean_pmdop{}_flat.npz'.format(test_AR,test_AR),allow_pickle=True) 
mag_flux = np.load('/home/jonas/Documents/AR_PREDICTION_MODELS/data/AR{}/mean_tiles9/mean_mag{}_flat.npz'.format(test_AR,test_AR),allow_pickle=True)
intensities = np.load('/home/jonas/Documents/AR_PREDICTION_MODELS/data/AR{}/mean_tiles9/mean_int{}_flat.npz'.format(test_AR,test_AR),allow_pickle=True) 
power_maps23 = power_maps['arr_0']
power_maps34 = power_maps['arr_1']
power_maps45 = power_maps['arr_2']
power_maps56 = power_maps['arr_3']
time = power_maps['arr_4']
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
inputs = np.concatenate([stacked_maps, mag_flux_reshaped], axis=1)
input_size = np.shape(inputs)[1]

# Initialize the LSTM and move it to GPU
lstm = LSTM(input_size, hidden_size, num_layers, num_pred).to(device)
saved_state_dict = torch.load(filename,map_location=device) # Load the state_dict you saved during training
new_state_dict = OrderedDict() # Create a new state_dict without the 'module.' prefix
for k, v in saved_state_dict.items():
    name = k[7:] if k.startswith('module.') else k  # remove 'module.' prefix
    new_state_dict[name] = v
lstm.load_state_dict(new_state_dict) # Load the corrected state_dict into your model
lstm.eval() # Set the model to evaluation mode

# Assuming prediction, y_test_tensors, ARs, learning_rate, and n_epochs are already defined
fig = plt.figure(figsize=(12, 10))  # Adjust the figure size if necessary
main_gs = gridspec.GridSpec(4, 2, figure=fig) # Create a GridSpec with 4 rows and 2 columns

# Loop to create 8 plots
starting_tile = 37-rid_of_top*9 # Change this CHANGEE
future = num_pred - 1
all_metrics = []
threshold = -0.01 #-0.006
sust_time = 4
before_plot = 40

for i in range(7):

    print(); print('Tile {}'.format(starting_tile+10+i))

    ### Validation
    X_test, y_test = lstm_ready(starting_tile+i,size,inputs,intensities,num_in,num_pred)#,min_p,max_p,min_i,max_i)
    X_test = X_test.to(device)
    all_predictions = lstm(X_test)
    pred = all_predictions[:, future].detach().cpu().numpy()
    true = y_test[:, future].numpy()
    last_known_idx = np.shape(intensities[starting_tile+i,:])[0]-np.shape(true)[0]-1 #the index in the timeline before we start predicting
    pred = recalibrate(pred,intensities[starting_tile+i,last_known_idx])
    first_pred_time = mdates.date2num(time[last_known_idx]-timedelta(hours=num_pred))

    ### Plot
    gs = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=main_gs[i], height_ratios=[4, 1, 1], hspace=0.05) # Define GridSpec for this iteration
    int_before_pred = intensities[starting_tile+i,last_known_idx-before_plot:last_known_idx]
    time_cut = time[last_known_idx-before_plot:last_known_idx+np.shape(pred)[0]]
    time_cut_mpl = mdates.date2num(time_cut) 
    nan_array = np.full(int_before_pred.shape, np.nan)
    zeros_array = np.full(int_before_pred.shape, 0)

    # Main plot
    ax0 = plt.subplot(gs[0])
    ax0.plot(time_cut_mpl, np.concatenate((nan_array, pred)))
    ax0.plot(time_cut_mpl, np.concatenate((int_before_pred, true)))
    ax0.plot(time_cut_mpl, smooth_with_numpy(np.concatenate((int_before_pred, true))), color = 'black', alpha = 0.25)
    ax0.set_ylabel(f'Tile {starting_tile+10+i}')  # Title for each plot (optional)
    #ax0.axvline(x=first_pred_time, color='darkturquoise', linestyle='--')
    ax0.axvline(x=NOAA_first_record, color='magenta', linestyle='--')  # Adjust color, linestyle, linewidth as needed 
    ax0.axvline(x=NOAA_second_record, color='darkmagenta', linestyle='--')
    #ax0.legend(['Observed','Predicted', 'Observed (Smooth)', 'First Prediction', r'NOAA $1^{st}$ Record', 'After Emergence'], fontsize = 7)  # Legend for each plot (optional)
    ax0.set_ylim([-0.1, 1.1])
    ax0.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5) # Enable the grid explicitly
    ax0.tick_params(axis='x', which='both', labelbottom=False)  # Hide x-axis tick labels
    ax0.xaxis_date() # Assuming ax0 should interpret x-axis values as dates
    ax0.xaxis.set_major_locator(mdates.DayLocator())  # Set major ticks to show once per day
    ax0.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Format the date
    plt.xticks(rotation=45, ha="right")  # Rotate x-tick labels for better readability

    # Subplot d_true
    ax1 = plt.subplot(gs[1])
    d_true = np.gradient(smooth_with_numpy(np.concatenate((int_before_pred, true))))#; d_true = smooth_with_numpy(d_true) # Assuming d_true is your data derivative
    dd_true = np.gradient(d_true)
    indicator_true = emergence_indication(d_true, threshold, sust_time) #emergence_indication2(dd_true) #emergence_indication(d_true, threshold, sust_time)
    first = True
    for j in range(len(d_true) - 1): # Now, plot using time_cut_mpl as x-values
        current_color = 'g' if indicator_true[j] == 0 else 'r'
        if current_color == 'r' and first == True: 
            readable_time = [mdates.num2date(time).strftime('%Y-%m-%d %H:%M:%S') for time in time_cut_mpl[j:j+2]]
            first = False; print('Observed First Emergence Time: {}'.format(readable_time[1]))
        ax1.plot(time_cut_mpl[j:j+2], d_true[j:j+2], color=current_color)  # Use time_cut_mpl for x-values
    #ax1.axvline(x=first_pred_time, color='darkturquoise', linestyle='--')
    ax1.axvline(x=NOAA_first_record, color='magenta', linestyle='--')  # Adjust color, linestyle, linewidth as needed
    ax1.axvline(x=NOAA_second_record, color='darkmagenta', linestyle='--')
    ax1.xaxis_date()  # Interpret x-axis values as dates
    ax1.xaxis.set_major_locator(mdates.DayLocator())  # Set major ticks to show once per day
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Format the date
    ax1.set_xticklabels([])
    ax1.set_ylim([-0.075, 0.075])
    ax1.set_yticks([0])
    ax1.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5)
    ax1.set_ylabel(r'$\frac{d Obs}{dt}$')

    # Subplot d_pred
    ax2 = plt.subplot(gs[2])
    d_pred = np.gradient(pred) #np.gradient(pred) # Assuming d_pred is your data derivative
    dd_pred = np.gradient(d_pred)
    indicator_pred = emergence_indication(d_pred, threshold, sust_time) #emergence_indication2(dd_pred) #emergence_indication(d_pred, threshold, sust_time)
    dd_pred = np.concatenate((zeros_array, dd_pred))
    d_pred = np.concatenate((zeros_array, d_pred))
    indicator_pred = np.concatenate((nan_array, indicator_pred))
    time_cut_mpl = mdates.date2num(time_cut)  # Convert datetime objects to Matplotlib dates
    first = True
    for k in range(len(dd_pred) - 1): # Now, plot using time_cut_mpl as x-values
        current_color = 'g' if indicator_pred[k] == 0 else 'r' if indicator_pred[k] == 1 else 'grey'
        if current_color == 'r' and first == True: 
            readable_time = [mdates.num2date(time).strftime('%Y-%m-%d %H:%M:%S') for time in time_cut_mpl[k:k+2]]
            first = False; print('Predicted First Emergence Time: {}'.format(readable_time[1]))
        alph = 1 if indicator_pred[k] in [0, 1] else 0
        ax2.plot(time_cut_mpl[k:k+2], d_pred[k:k+2], color=current_color, alpha=alph)  # Use time_cut_mpl for x-values
    #ax2.axvline(x=first_pred_time, color='darkturquoise', linestyle='--')
    ax2.axvline(x=NOAA_first_record, color='magenta', linestyle='--')  # Adjust color, linestyle, linewidth as needed
    ax2.axvline(x=NOAA_second_record, color='darkmagenta', linestyle='--')
    ax2.xaxis_date()  # Tell Matplotlib to interpret the x-axis values as dates
    ax2.xaxis.set_major_locator(mdates.DayLocator())  # Set major ticks to show once per day
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%y')) # Format the date
    ax2.tick_params(axis='x', which='major', labelsize=9)  # Adjust 'labelsize' as needed
    ax2.set_ylim([-0.075, 0.075])
    ax2.set_yticks([0])  # Set the y-axis to only have a tick at 0
    ax2.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5)
    ax2.set_ylabel(r'$\frac{d Pred}{dt}$')

    # Evaluation metrics
    metrics = calculate_metrics(true, pred)
    all_metrics.append(metrics)
    #print(f"MAE: {metrics[0]}")
    #print(f"MSE: {metrics[1]}")
    print(f"RMSE: {metrics[2]}")
    #print(f"RMSLE: {metrics[3]}")
    #print(f"R-squared: {metrics[4]}")

# Last subplot with continuum intensities
plt.subplot(4, 2, 8)
gs_last = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=main_gs[7], wspace=0.05) # Create a GridSpec for the last subplot area with 1 row and 2 columns
ax_image1 = plt.subplot(gs_last[0, 0]) # Plot the first image
ax_image1.imshow(NOAA_first_int_map, cmap='gray')
add_grid_lines(ax_image1)  # Add grid lines to the first image
for tile_num in range(starting_tile, starting_tile + 7): highlight_tile(ax_image1, tile_num+10) # Loop to highlight tiles from starting_tile to starting_tile + 7
ax_image1.set_xlabel('{}'.format(NOAA_first.strftime('%d/%m/%y %H:%M')))
ax_image1.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelleft=False, labelbottom=True)     
ax_image1.set_xticks([]); ax_image1.set_yticks([])
ax_image2 = plt.subplot(gs_last[0, 1]) # Plot the second image
ax_image2.imshow(NOAA_second_int_map, cmap='gray')
add_grid_lines(ax_image2)  # Add grid lines to the first image
for tile_num in range(starting_tile, starting_tile + 7): highlight_tile(ax_image2, tile_num+10) # Loop to highlight tiles from starting_tile to starting_tile + 7
ax_image1.set_title('Continuum Intensity', fontsize=10) 
ax_image2.set_title('Continuum Intensity', fontsize=10) 
ax_image2.set_xlabel('{}'.format(NOAA_second.strftime('%d/%m/%y %H:%M')))
ax_image2.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelleft=False, labelbottom=True)     
ax_image2.set_xticks([]); ax_image2.set_yticks([])

# Print the metrics at the bottom
all_metrics_np = np.array(all_metrics) # Convert all_metrics to a NumPy array for easier manipulation
means = np.mean(all_metrics_np, axis=0) # Calculate the mean and standard deviation for each metric across the 7 runs
stds = np.std(all_metrics_np, axis=0)
mae_string = r'Average metrics for all tiles plotted:  $\mathrm{{MAE}} = {}$,  $\mathrm{{MSE}} = {}$,  $\mathrm{{RMSE}} = {}$,  $\mathrm{{RMSLE}} = {}$,  $R^2 = {}$'.format(round(means[0],3), round(means[1],3), round(means[2],3), round(means[3],3), round(means[4],3))
fig.text(0.5, 0.02, mae_string, ha='center', va='bottom', fontsize=10)

plt.tight_layout()  # Adjusts subplot parameters for better layout
plt.subplots_adjust(top=0.96, bottom = 0.075)  # Adjust top spacing
plt.suptitle('LSTM Results for AR{} (TW = {}h, RoT = {}, In = {})'.format(test_AR,num_pred,rid_of_top,num_in), y=0.99)
save_path = './ApJ_paper_lstm/best/AR{}_{}.pdf'.format(test_AR,os.path.splitext(os.path.basename(pth_file[0]))[0])
plt.savefig(save_path); print('Results saved at: ' + save_path)