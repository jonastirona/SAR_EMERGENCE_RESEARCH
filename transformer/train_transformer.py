import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import argparse
import time
import os

from transformer_model import SARTransformerLocalTile
from data_loader_transformer import load_ar_data_enhanced, cross_ar_tile_data_preparation_attention

def emergence_aware_loss(predictions, targets, lambda_emergence=0.1):
    """
    Loss function that encourages attention to emergence patterns
    """
    # Standard MSE loss
    mse_loss = nn.MSELoss()(predictions, targets)
    
    # Emergence pattern loss - encourage learning of derivative patterns
    pred_derivatives = torch.gradient(predictions, dim=1)[0]
    target_derivatives = torch.gradient(targets, dim=1)[0]
    derivative_loss = nn.MSELoss()(pred_derivatives, target_derivatives)
    
    # Temporal consistency loss
    temporal_consistency = torch.mean(torch.abs(pred_derivatives[:, 1:] - pred_derivatives[:, :-1]))
    
    total_loss = mse_loss + lambda_emergence * derivative_loss + 0.01 * temporal_consistency
    
    return total_loss

def train_transformer_with_full_attention(
    data_path,
    output_dir,
    ARs,
    test_AR,
    num_pred=12,
    rid_of_top=1, 
    num_in=128,
    num_layers=6,
    hidden_size=256,
    n_epochs=1000,
    learning_rate=0.001,
    nhead=8,
    dropout=0.1
):
    """
    Train transformer using full attention capabilities
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on: {device}')
    print(f"Parameters: pred={num_pred}, layers={num_layers}, hidden={hidden_size}")
    
    # Setup
    size = 9
    tiles = size**2 - 2*size*rid_of_top
    ARs_full = ARs + [test_AR]
    
    # Load data
    print('Loading data for full attention transformer...')
    all_inputs, all_intensities = load_ar_data_enhanced(
        ARs_full, rid_of_top, size, num_in, num_pred, data_path
    )
    
    input_size = all_inputs.shape[1]  # Should be 5 (4 power + 1 flux)
    print(f"Input features: {input_size}")
    
    # Initialize enhanced transformer model
    model = SARTransformerLocalTile(
        input_dim=input_size,
        d_model=hidden_size,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout,
        output_len=num_pred,
        max_seq_len=num_in + 50  # Extra capacity
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Enhanced Transformer parameters: {total_params:,}")
    
    # Enhanced optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=learning_rate, epochs=n_epochs, 
        steps_per_epoch=tiles, pct_start=0.1
    )
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, "enhanced_transformer_training_results.txt")
    
    print(f"\n=== TRAINING ENHANCED TRANSFORMER (FULL ATTENTION) ===")
    
    with open(results_file, "w") as file:
        file.write("Enhanced Transformer Training Results (Full Attention)\n")
        file.write(f"Model: d_model={hidden_size}, layers={num_layers}, heads={nhead}\n")
        file.write(f"Data: {len(ARs_full)} ARs, {tiles} tiles\n\n")
        
        best_loss = float('inf')
        
        for epoch in range(n_epochs):
            model.train()
            epoch_losses = []
            
            # Train on all tiles in each epoch
            for tile in range(0, tiles, 8):  # Sample tiles for speed
                # Get full attention data for this tile
                X_tile, y_tile = cross_ar_tile_data_preparation_attention(
                    tile, size, all_inputs, all_intensities, num_in, num_pred
                )
                
                if len(X_tile) == 0:
                    continue
                
                X_tile = X_tile.to(device)
                y_tile = y_tile.to(device)
                
                # Forward pass
                optimizer.zero_grad()
                predictions = model(X_tile)
                
                # Enhanced loss function
                loss = emergence_aware_loss(predictions, y_tile, lambda_emergence=0.1)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                
                epoch_losses.append(loss.item())
            
            avg_loss = np.mean(epoch_losses) if epoch_losses else float('inf')
            
            if epoch % 50 == 0:
                print(f"Epoch {epoch}: loss={avg_loss:.6f}, lr={scheduler.get_last_lr()[0]:.6f}")
                file.write(f"{epoch}, {avg_loss:.6f}, {scheduler.get_last_lr()[0]:.6f}\n")
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_path = os.path.join(output_dir, f"best_enhanced_transformer.pth")
                torch.save(model.state_dict(), best_model_path)
    
    # Save final model
    model_path = os.path.join(output_dir, f"enhanced_transformer_t{num_pred}_r{rid_of_top}_i{num_in}_n{num_layers}_h{hidden_size}_e{n_epochs}_l{learning_rate}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Enhanced model saved: {model_path}")
    
    return model_path

def main():
    parser = argparse.ArgumentParser(description='Train Enhanced Attention Transformer')
    parser.add_argument('--data_path', type=str, default='/mmfs1/project/mx6/jst26/SAR_EMERGENCE_RESEARCH/data', help='Data directory path')
    parser.add_argument('--output_dir', type=str, default='./enhanced_transformer_results', help='Output directory')
    parser.add_argument('--num_pred', type=int, default=12, help='Prediction window')
    parser.add_argument('--rid_of_top', type=int, default=1, help='Trim top/bottom rows')
    parser.add_argument('--num_in', type=int, default=128, help='Input sequence length (longer for attention)')
    parser.add_argument('--num_layers', type=int, default=6, help='Transformer layers')
    parser.add_argument('--hidden_size', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--n_epochs', type=int, default=1000, help='Training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--nhead', type=int, default=8, help='Attention heads')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    args = parser.parse_args()
    
    # Same ARs as LSTM training
    ARs = [11130, 11149, 11158, 11162, 11199, 11327, 11344, 11387, 11393, 11416, 
           11422, 11455, 11619, 11640, 11660, 11678, 11682, 11765, 11768, 11776, 
           11916, 11928, 12036, 12051, 12085, 12089, 12144, 12175, 12203, 12257, 
           12331, 12494, 12659, 12778, 12864, 12877, 12900, 12929, 13004, 13085, 13098]
    
    test_AR = 13179
    
    print("="*70)
    print("ENHANCED TRANSFORMER TRAINING (FULL ATTENTION)")
    print("="*70)
    print("Key Features:")
    print("- Multi-head attention with relative positional encoding")
    print("- Emergence-aware loss function")
    print("- Multi-scale local pattern extraction")
    print("- Learnable positional embeddings")
    print("- Full sequence attention (no sliding windows)")
    print("="*70)
    
    start_time = time.time()
    
    model_path = train_transformer_with_full_attention(
        data_path=args.data_path,
        output_dir=args.output_dir,
        ARs=ARs,
        test_AR=test_AR,
        num_pred=args.num_pred,
        rid_of_top=args.rid_of_top,
        num_in=args.num_in,
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        nhead=args.nhead,
        dropout=args.dropout
    )
    
    end_time = time.time()
    print(f"\nTraining completed in {(end_time - start_time)/60:.2f} minutes")
    print(f"Model saved: {model_path}")

if __name__ == '__main__':
    main()