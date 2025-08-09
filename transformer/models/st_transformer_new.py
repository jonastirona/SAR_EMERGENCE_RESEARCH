import torch
import torch.nn as nn

class SpatioTemporalTransformer(nn.Module):
    def __init__(self, input_dim, max_seq_len, embed_dim, num_heads, ff_dim, num_layers, output_dim, dropout,
                 use_pre_mlp_norm=True):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        self.use_pre_mlp_norm = use_pre_mlp_norm

        self.embedding = nn.Linear(input_dim, embed_dim)
        pe = self._generate_positional_encoding(max_seq_len, embed_dim)
        self.register_buffer('positional_encoding', pe)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        if self.use_pre_mlp_norm:
            self.pre_mlp_norm = nn.LayerNorm(embed_dim)

        self.mlp_head = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, output_dim)
        )

    def _generate_positional_encoding(self, seq_len, dim):
        pe = torch.zeros(1, seq_len, dim)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim))
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x, src_key_padding_mask=None):
        seq_len = x.size(1)
        x = self.embedding(x)
        x = x + self.positional_encoding[:, :seq_len, :]

        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        if src_key_padding_mask is not None:
            mask = ~src_key_padding_mask.unsqueeze(-1).expand(x.shape).bool()
            x = (x * mask).sum(dim=1)
            num_valid_tokens = mask.sum(dim=1)
            num_valid_tokens = torch.clamp(num_valid_tokens, min=1)
            x = x / num_valid_tokens
        else:
            x = x.mean(dim=1)

        if self.use_pre_mlp_norm:
            x = self.pre_mlp_norm(x)

        out = self.mlp_head(x)
        return out

if __name__ == "__main__":
    model = SpatioTemporalTransformer(
        input_dim=5,
        max_seq_len=200,
        embed_dim=64,
        num_heads=8,
        ff_dim=128,
        num_layers=2,
        output_dim=12,
        dropout=0.1,
        use_pre_mlp_norm=True
    )
    x1 = torch.randn(80, 5)
    x2 = torch.randn(110, 5)
    from torch.nn.utils.rnn import pad_sequence
    padded_batch = pad_sequence([x1, x2], batch_first=True, padding_value=0.0)
    mask = torch.zeros(2, 110, dtype=torch.bool)
    mask[0, 80:] = True
    y_pred = model(padded_batch, src_key_padding_mask=mask)
    print(f"Input shape: {padded_batch.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Output shape: {y_pred.shape}")