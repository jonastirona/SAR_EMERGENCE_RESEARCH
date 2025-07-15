import torch
from ns_Transformer import Model

# Minimal config for the model
def get_minimal_config():
    class Config:
        pred_len = 4
        seq_len = 8
        label_len = 4
        output_attention = False
        enc_in = 3
        dec_in = 3
        d_model = 8
        embed = 'fixed'
        freq = 'h'
        dropout = 0.1
        n_heads = 2
        d_ff = 16
        activation = 'gelu'
        factor = 1
        e_layers = 1
        d_layers = 1
        c_out = 1
        p_hidden_dims = [8, 8]
        p_hidden_layers = 2
    return Config()

if __name__ == "__main__":
    config = get_minimal_config()
    model = Model(config)
    B = 2  # batch size
    S = config.seq_len
    E = config.enc_in
    L = config.label_len
    P = config.pred_len
    # Dummy input: [batch, seq_len, enc_in]
    x_enc = torch.randn(B, S, E)
    # Correct time feature ranges
    month = torch.randint(0, 13, (B, S, 1))      # 0-12
    day = torch.randint(0, 32, (B, S, 1))        # 0-31
    weekday = torch.randint(0, 7, (B, S, 1))     # 0-6
    hour = torch.randint(0, 24, (B, S, 1))       # 0-23
    x_mark_enc = torch.cat([month, day, weekday, hour], dim=2)
    # Repeat for decoder
    month = torch.randint(0, 13, (B, L+P, 1))
    day = torch.randint(0, 32, (B, L+P, 1))
    weekday = torch.randint(0, 7, (B, L+P, 1))
    hour = torch.randint(0, 24, (B, L+P, 1))
    x_mark_dec = torch.cat([month, day, weekday, hour], dim=2)
    x_dec = torch.randn(B, L + P, E)
    out = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
    print("Output shape:", out.shape)
    print("Test passed!") 