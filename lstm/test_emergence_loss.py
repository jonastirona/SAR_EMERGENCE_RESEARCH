"""Unit test for emergence-aware loss functions.

Verifies:
1. differentiable_smooth produces correct shape and smooths
2. soft_emergence_time finds onset on a synthetic ramp signal
3. emergence_timing_loss computes correctly and gradients flow
4. Quiet samples contribute 0 to timing loss
"""

import torch
import numpy as np
import sys
import os

# Add parent directory to path so we can import functions
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from functions import (
    differentiable_smooth,
    soft_emergence_time,
    emergence_timing_loss,
)


def test_differentiable_smooth():
    """Test that smoothing preserves shape and actually smooths."""
    print("--- test_differentiable_smooth ---")
    batch, T = 4, 20
    x = torch.randn(batch, T)

    smoothed = differentiable_smooth(x, window_size=5)
    assert smoothed.shape == x.shape, f"Shape mismatch: {smoothed.shape} vs {x.shape}"

    # Smoothed signal should have lower variance
    assert smoothed.std() <= x.std() + 0.01, "Smoothing did not reduce variance"

    # Window size 1 should return identity
    identity = differentiable_smooth(x, window_size=1)
    assert torch.allclose(identity, x), "Window size 1 should be identity"

    print("  PASSED: shape preserved, variance reduced, identity at window=1")


def test_soft_emergence_time_synthetic():
    """Test onset detection on a known synthetic signal."""
    print("--- test_soft_emergence_time_synthetic ---")

    # Create a signal that ramps up at timestep 8 out of 20
    # Derivative is near-zero before t=8, then jumps above threshold
    T = 20
    signal = torch.zeros(1, T)
    signal[0, :8] = 0.0  # flat
    signal[0, 8:] = 0.03  # above threshold=0.01 (these are derivative values)

    # Use high k for near-hard threshold
    soft_time, has_emg = soft_emergence_time(
        signal, threshold=0.01, k=100.0, window_size=1
    )

    print(
        f"  Detected onset time: {soft_time.item():.2f} (expected ~7, rising edge at index 7)"
    )
    print(f"  Has emergence: {has_emg.item():.4f} (expected ~1.0)")

    # The rising edge should be near index 7 (transition from 0 to 0.03)
    # Note: rising edge is computed on the diff of soft_ind, so it's shifted by 1
    assert abs(soft_time.item() - 7.0) < 2.0, (
        f"Onset time too far from expected: {soft_time.item()}"
    )
    assert has_emg.item() > 0.5, f"Should detect emergence: {has_emg.item()}"

    print("  PASSED: onset detected near expected time")


def test_emergence_timing_loss_gradient_flow():
    """Test that gradients flow through the emergence timing loss."""
    print("--- test_emergence_timing_loss_gradient_flow ---")

    batch = 8
    T = 12

    # Create predictions that require gradients
    pred = torch.randn(batch, T, requires_grad=True)
    y = torch.randn(batch, T)

    loss = emergence_timing_loss(pred, y, threshold=0.01, k=20.0)

    # Loss should be a scalar
    assert loss.dim() == 0, f"Loss should be scalar, got shape {loss.shape}"

    # Gradients should flow
    loss.backward()
    assert pred.grad is not None, "Gradient did not flow to predictions"
    assert not torch.all(pred.grad == 0), "All gradients are zero — no signal"

    print(f"  Loss value: {loss.item():.6f}")
    print(f"  Gradient norm: {pred.grad.norm().item():.6f}")
    print("  PASSED: gradients flow correctly")


def test_quiet_samples():
    """Test that quiet samples (no emergence) contribute 0 to timing loss."""
    print("--- test_quiet_samples ---")

    batch = 4
    T = 12

    # Create flat signals (derivative ~ 0, well below threshold)
    pred = torch.ones(batch, T) * 0.5  # constant → derivative = 0
    pred.requires_grad_(True)
    y = torch.ones(batch, T) * 0.5

    loss = emergence_timing_loss(pred, y, threshold=0.01, k=50.0)

    print(f"  Loss for quiet samples: {loss.item():.8f} (expected ~0)")
    assert loss.item() < 0.01, f"Quiet sample loss should be near 0, got {loss.item()}"

    print("  PASSED: quiet samples contribute ~0 loss")


def test_aligned_emergence_low_loss():
    """Test that perfectly aligned emergences produce near-zero loss."""
    print("--- test_aligned_emergence_low_loss ---")

    T = 20
    # Both signals have same derivative pattern
    signal = torch.zeros(1, T)
    for i in range(T):
        if i < 8:
            signal[0, i] = 0.0
        else:
            signal[0, i] = 0.03 * (i - 7)  # ramp up starting at t=8

    pred = signal.clone().requires_grad_(True)
    y = signal.clone()

    loss = emergence_timing_loss(pred, y, threshold=0.01, k=50.0)

    print(f"  Loss for aligned emergence: {loss.item():.8f} (expected ~0)")
    assert loss.item() < 0.1, (
        f"Aligned emergence should have near-zero loss, got {loss.item()}"
    )

    print("  PASSED: aligned emergences produce ~0 loss")


def test_misaligned_emergence_high_loss():
    """Test that misaligned emergences produce non-zero loss."""
    print("--- test_misaligned_emergence_high_loss ---")

    T = 20
    # Prediction: emergence at t=5
    pred_signal = torch.zeros(1, T)
    pred_signal[0, 5:] = torch.linspace(0, 0.5, T - 5)

    # Truth: emergence at t=12
    true_signal = torch.zeros(1, T)
    true_signal[0, 12:] = torch.linspace(0, 0.5, T - 12)

    pred_signal.requires_grad_(True)

    loss = emergence_timing_loss(pred_signal, true_signal, threshold=0.01, k=50.0)

    print(f"  Loss for misaligned emergence: {loss.item():.4f} (expected > 0)")
    assert loss.item() > 0.1, (
        f"Misaligned emergence should have non-zero loss, got {loss.item()}"
    )

    # Verify gradients exist
    loss.backward()
    assert pred_signal.grad is not None, "No gradients for misaligned case"

    print("  PASSED: misaligned emergences produce significant loss")


if __name__ == "__main__":
    print("=" * 60)
    print("Emergence-Aware Loss Unit Tests")
    print("=" * 60)

    test_differentiable_smooth()
    test_soft_emergence_time_synthetic()
    test_emergence_timing_loss_gradient_flow()
    test_quiet_samples()
    test_aligned_emergence_low_loss()
    test_misaligned_emergence_high_loss()

    print()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
