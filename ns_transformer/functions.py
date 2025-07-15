import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add project root and ns_transformer to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'ns_transformer', 'ns_models'))

# Import Non-stationary Transformer
from ns_models.ns_Transformer import Model as NSTransformer


def get_ns_transformer_config():
    class Config:
        pred_len = 12
        seq_len = 110
        label_len = 12
        output_attention = False
        enc_in = 5
        dec_in = 5
        d_model = 64
        embed = 'fixed'
        freq = 'h'
        dropout = 0.1
        n_heads = 4
        d_ff = 128
        activation = 'gelu'
        factor = 1
        e_layers = 3
        d_layers = 3
        c_out = 1
        p_hidden_dims = [64, 64]
        p_hidden_layers = 2
    return Config()


# Create a wrapper class to make NSTransformer compatible with simple interface
class NSTransformerWrapper(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ns_model = NSTransformer(config)
        self.config = config
        
    def forward(self, x):
        # Create proper time features for NSTransformer
        batch_size, seq_len, _ = x.shape
        
        # Create time features in the format expected by TemporalEmbedding
        # x_mark should have shape (batch, seq_len, 5) with [month, day, weekday, hour, minute]
        positions = torch.arange(seq_len, device=x.device).float()
        
        # Map positions to time-like features (0-1 range then scale)
        normalized_positions = positions / max(seq_len - 1, 1)
        
        # Create time features based on position
        month = (normalized_positions * 12).long().clamp(0, 12)
        day = (normalized_positions * 31).long().clamp(0, 31)
        weekday = (normalized_positions * 6).long().clamp(0, 6)
        hour = (normalized_positions * 23).long().clamp(0, 23)
        minute = torch.zeros_like(month)  # Default to 0 for minute
        
        # Stack features: [month, day, weekday, hour, minute]
        x_mark_enc = torch.stack([month, day, weekday, hour, minute], dim=1)  # (seq_len, 5)
        x_mark_enc = x_mark_enc.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, seq_len, 5)
        
        # Create decoder input with proper structure
        # The NSTransformer expects: [label_len + pred_len] total decoder length
        total_dec_len = self.config.label_len + self.config.pred_len
        x_dec = torch.zeros(batch_size, total_dec_len, x.shape[-1], device=x.device)
        
        # Create decoder time features for the full decoder length
        dec_positions = torch.arange(seq_len, seq_len + total_dec_len, device=x.device).float()
        dec_normalized = dec_positions / max(seq_len + total_dec_len - 1, 1)
        
        dec_month = (dec_normalized * 12).long().clamp(0, 12)
        dec_day = (dec_normalized * 31).long().clamp(0, 31)
        dec_weekday = (dec_normalized * 6).long().clamp(0, 6)
        dec_hour = (dec_normalized * 23).long().clamp(0, 23)
        dec_minute = torch.zeros_like(dec_month)
        
        x_mark_dec = torch.stack([dec_month, dec_day, dec_weekday, dec_hour, dec_minute], dim=1)
        x_mark_dec = x_mark_dec.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Forward pass through NSTransformer
        output = self.ns_model(x, x_mark_enc, x_dec, x_mark_dec)
        
        # The NSTransformer outputs all features due to de-normalization
        # We need to extract only the intensity predictions (last feature)
        # output shape: (batch, pred_len, 5) -> (batch, pred_len)
        intensity_predictions = output[:, -self.config.pred_len:, -1]  # Take only the last feature (intensity)
        
        return intensity_predictions 