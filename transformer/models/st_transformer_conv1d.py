import torch
import torch.nn as nn


class SpatioTemporalTransformer(nn.Module):
    def __init__(self, input_dim, seq_len, embed_dim, num_heads, ff_dim, num_layers, output_dim, dropout, 
                 use_cls_token=True, use_attention_pool=True, use_pre_mlp_norm=True, use_temporal_conv=False):
        super(SpatioTemporalTransformer, self).__init__()
        
        # Validate parameters
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        
        self.use_cls_token = use_cls_token
        self.use_attention_pool = use_attention_pool
        self.use_pre_mlp_norm = use_pre_mlp_norm
        self.use_temporal_conv = use_temporal_conv
        
        # Temporal convolution layers for emergence pattern detection
        if self.use_temporal_conv:
            self.temporal_conv = nn.Sequential(
                # First conv layer: detect local temporal patterns (7-hour window)
                nn.Conv1d(input_dim, embed_dim//2, kernel_size=7, padding=3),
                nn.ReLU(),
                nn.Dropout(dropout),
                # Second conv layer: refine patterns (5-hour window)  
                nn.Conv1d(embed_dim//2, embed_dim, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            # Optional: additional projection layer for dimensionality consistency
            self.conv_projection = nn.Linear(embed_dim, embed_dim)
        else:
            # Original embedding layer
            self.embedding = nn.Linear(input_dim, embed_dim)
        
        self.positional_encoding = self._generate_positional_encoding(seq_len, embed_dim)
        
        # Add cls_token if used
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Add attention pooling if used
        if self.use_attention_pool:
            self.attention_pool = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        # Add pre_mlp_norm if requested (enable by default for new models)
        if self.use_pre_mlp_norm:
            self.pre_mlp_norm = nn.LayerNorm(embed_dim)

        # Update MLP head to match saved model structure
        self.mlp_head = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),           # mlp_head.0
            nn.ReLU(),                              # mlp_head.1
            nn.Dropout(dropout),                    # mlp_head.2
            nn.Linear(ff_dim, ff_dim),              # mlp_head.3
            nn.ReLU(),                              # mlp_head.4
            nn.Dropout(dropout),                    # mlp_head.5
            nn.Linear(ff_dim, output_dim)           # mlp_head.6
        )

    def _generate_positional_encoding(self, seq_len, dim):
        pe = torch.zeros(seq_len, dim)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, seq_len, dim)

    def forward(self, x):
        # Input shape: (batch_size, seq_len, input_dim)
        
        if self.use_temporal_conv:
            # Apply temporal convolution for local pattern detection
            # Transpose for Conv1d: (batch_size, input_dim, seq_len)
            x_conv = x.transpose(1, 2)
            
            # Apply temporal convolution layers
            x_conv = self.temporal_conv(x_conv)
            
            # Transpose back: (batch_size, seq_len, embed_dim)
            x = x_conv.transpose(1, 2)
            
            # Optional projection for consistency
            x = self.conv_projection(x)
        else:
            # Original embedding approach
            x = self.embedding(x)
        
        # Add positional encoding
        x = x + self.positional_encoding.to(x.device)
        
        # Add cls_token if used
        if self.use_cls_token:
            batch_size = x.size(0)
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)
        
        # Use attention pooling or mean pooling
        if self.use_attention_pool:
            query = x.mean(dim=1, keepdim=True)  # Global average as query
            pooled_output, _ = self.attention_pool(query, x, x)
            x = pooled_output.squeeze(1)
        else:
            x = x.mean(dim=1)
        
        # Apply pre-MLP normalization only if the layer exists
        if self.use_pre_mlp_norm:
            x = self.pre_mlp_norm(x)
            
        out = self.mlp_head(x)
        return out
    
if __name__ == "__main__":
    # Example usage with temporal convolution enabled
    model = SpatioTemporalTransformer(
        input_dim=5, 
        seq_len=110, 
        embed_dim=128, 
        num_heads=4, 
        ff_dim=256, 
        num_layers=3, 
        output_dim=12,
        dropout=0.1,
        use_pre_mlp_norm=True,
        use_temporal_conv=True  # Enable temporal convolution
    )
    X = torch.randn(32, 110, 5)
    y_pred = model(X)  # -> (32, 12)
    print(f"Input shape: {X.shape}")
    print(f"Output shape: {y_pred.shape}")