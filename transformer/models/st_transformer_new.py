import torch
import torch.nn as nn

class SpatioTemporalTransformer(nn.Module):
    def __init__(self, input_dim, seq_len, embed_dim, num_heads, ff_dim, num_layers, output_dim, dropout, use_pre_mlp_norm=True):
        """
        Spatio-Temporal Transformer for SAR emergence prediction.
        
        Args:
            input_dim (int): Number of input features (e.g., 5 for 4 power maps + 1 magnetic flux)
            seq_len (int): Input sequence length 
            embed_dim (int): Embedding dimension
            num_heads (int): Number of attention heads
            ff_dim (int): Feed-forward dimension
            num_layers (int): Number of transformer layers
            output_dim (int): Number of output predictions
            dropout (float, optional): Dropout probability
            use_pre_mlp_norm (bool, optional): Whether to include pre-MLP layer normalization
        """
        super(SpatioTemporalTransformer, self).__init__()
        
        # Validate parameters
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        
        self.use_pre_mlp_norm = use_pre_mlp_norm
        
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.positional_encoding = self._generate_positional_encoding(seq_len, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Only create pre_mlp_norm if requested
        if self.use_pre_mlp_norm:
            self.pre_mlp_norm = nn.LayerNorm(embed_dim)

        self.mlp_head = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, output_dim)
        )

    def _generate_positional_encoding(self, seq_len, dim):
        pe = torch.zeros(seq_len, dim)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, seq_len, dim)

    def forward(self, x):  # x: (batch, seq_len, input_dim)
        x = self.embedding(x)  # -> (batch, seq_len, embed_dim)
        x = x + self.positional_encoding.to(x.device)  # Add positional encoding
        x = self.transformer_encoder(x)  # -> (batch, seq_len, embed_dim)

        x = x.mean(dim=1)  # Global average pooling over time
        
        # Apply pre-MLP normalization only if the layer exists
        if self.use_pre_mlp_norm:
            x = self.pre_mlp_norm(x)
            
        out = self.mlp_head(x)  # -> (batch, output_dim)
        return out
    
if __name__ == "__main__":
    # Example usage with explicit parameters
    model = SpatioTemporalTransformer(
        input_dim=5, 
        seq_len=110, 
        embed_dim=64, 
        num_heads=8, 
        ff_dim=128, 
        num_layers=2, 
        output_dim=12,
        dropout=0.1,
        use_pre_mlp_norm=True
    )
    X = torch.randn(120, 110, 5)
    y_pred = model(X)  # -> (120, 12)
    print(y_pred.shape)