import torch
import torch.nn as nn

class SpatioTemporalTransformer(nn.Module):
    def __init__(self, input_dim=5, seq_len=110, embed_dim=408, num_heads=8, ff_dim=2984, num_layers=7, output_dim=12, dropout=0.3973226820560444):
        super(SpatioTemporalTransformer, self).__init__()
        
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
        x = self.pre_mlp_norm(x)  # Apply layer norm before MLP head
        out = self.mlp_head(x)  # -> (batch, output_dim)
        return out
    
if __name__ == "__main__":
    model = SpatioTemporalTransformer(input_dim=5, seq_len=110)
    X = torch.randn(120, 110, 5)
    y_pred = model(X)  # -> (120, 12)
    print(y_pred.shape)