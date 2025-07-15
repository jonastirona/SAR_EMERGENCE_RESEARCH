#!/usr/bin/env python3
"""
Debug script to test NSTransformerWrapper
"""

import torch
import sys
import os

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'ns_transformer', 'ns_models'))

def test_wrapper():
    """Test the NSTransformerWrapper with debug output"""
    
    # Create a simple config
    class Config:
        pred_len = 12
        seq_len = 110
        label_len = 12
        output_attention = False
        enc_in = 5
        dec_in = 5
        c_out = 1  # Only predict intensity
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
        p_hidden_dims = [64, 64]
        p_hidden_layers = 2
    
    config = Config()
    
    print("Config:")
    print(f"  seq_len: {config.seq_len}")
    print(f"  label_len: {config.label_len}")
    print(f"  pred_len: {config.pred_len}")
    print(f"  enc_in: {config.enc_in}")
    print(f"  dec_in: {config.dec_in}")
    print(f"  c_out: {config.c_out}")
    
    # Create wrapper
    from train_ns_transformer_w_lstm_architecture import NSTransformerWrapper
    wrapper = NSTransformerWrapper(config)
    
    # Create test input
    batch_size = 2
    seq_len = 110
    input_dim = 5
    
    x = torch.randn(batch_size, seq_len, input_dim)
    
    print(f"\nInput shape: {x.shape}")
    
    # Test forward pass
    try:
        output = wrapper(x)
        print(f"✓ Forward pass successful!")
        print(f"Output shape: {output.shape}")
        print(f"Expected output shape: ({batch_size}, {config.pred_len})")
        
        if output.shape == (batch_size, config.pred_len):
            print("✓ Output shape matches expected!")
        else:
            print("✗ Output shape mismatch!")
            print(f"Expected: ({batch_size}, {config.pred_len})")
            print(f"Got: {output.shape}")
            
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_wrapper()
    if success:
        print("\n🎉 NSTransformerWrapper debug test passed!")
    else:
        print("\n❌ NSTransformerWrapper debug test failed!") 