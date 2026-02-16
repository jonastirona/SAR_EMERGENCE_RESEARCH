"""Unit test for the 'Lazy Student' fix in emergence timing scoring.

Verifies that:
1. Flat-line (lazy) models get penalized with max MAE, not 0
2. Good models that detect emergence get lower MAE
3. Detection rate = 0 for flat-line, 1.0 for correct predictors
4. No true emergence → MAE = 0, detection rate = 1.0
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from functions import emergence_indication


def compute_emergence_metrics(all_preds_np, all_y_np):
    """
    Extracted emergence timing logic from validate_model for isolated testing.
    Mirrors the exact logic in functions.py validate_model.
    """
    threshold = 0.01
    sust_time = 4
    num_pred = all_preds_np.shape[1]
    timing_errors = []
    true_emergence_count = 0
    detected_count = 0
    missed_count = 0
    false_alarm_count = 0

    for i in range(len(all_preds_np)):
        d_pred_i = np.gradient(all_preds_np[i])
        d_true_i = np.gradient(all_y_np[i])
        ind_pred = emergence_indication(d_pred_i, threshold, sust_time)
        ind_true = emergence_indication(d_true_i, threshold, sust_time)

        t_pred_i = None
        t_true_i = None
        for t_idx in range(len(ind_pred)):
            if ind_pred[t_idx] == 1 and t_pred_i is None:
                t_pred_i = t_idx
            if ind_true[t_idx] == 1 and t_true_i is None:
                t_true_i = t_idx
            if t_pred_i is not None and t_true_i is not None:
                break

        if t_true_i is not None:
            true_emergence_count += 1
            if t_pred_i is not None:
                timing_errors.append(abs(t_pred_i - t_true_i))
                detected_count += 1
            else:
                timing_errors.append(num_pred)
                missed_count += 1
        else:
            if t_pred_i is not None:
                false_alarm_count += 1

    emergence_timing_mae = np.mean(timing_errors) if timing_errors else 0.0
    detection_rate = (
        detected_count / true_emergence_count if true_emergence_count > 0 else 1.0
    )
    false_alarm_rate = (
        false_alarm_count / len(all_preds_np) if len(all_preds_np) > 0 else 0.0
    )

    return {
        "MAE": emergence_timing_mae,
        "Detection_Rate": detection_rate,
        "False_Alarm_Rate": false_alarm_rate,
        "True_Emergences": true_emergence_count,
        "Detected": detected_count,
        "Missed": missed_count,
        "False_Alarms": false_alarm_count,
    }


def test_lazy_model_penalized():
    """A flat-line model should get MAE = num_pred, NOT 0."""
    print("--- test_lazy_model_penalized ---")
    num_pred = 12

    # Truth: clear emergence (ramp up starting at step 4)
    truth = np.zeros((5, num_pred))
    for i in range(5):
        truth[i, 4:] = np.linspace(0, 0.5, num_pred - 4)

    # Lazy model: flat line (predicts nothing)
    lazy_preds = np.ones((5, num_pred)) * 0.5  # constant

    metrics = compute_emergence_metrics(lazy_preds, truth)

    print(f"  Lazy MAE: {metrics['MAE']:.1f} (expected: {num_pred})")
    print(f"  Detection Rate: {metrics['Detection_Rate']:.2f} (expected: 0.0)")
    print(f"  Missed: {metrics['Missed']} / {metrics['True_Emergences']}")

    assert metrics["MAE"] == num_pred, (
        f"Lazy model should get MAE={num_pred}, got {metrics['MAE']}"
    )
    assert metrics["Detection_Rate"] == 0.0, (
        f"Lazy model should have 0 detection rate, got {metrics['Detection_Rate']}"
    )
    assert metrics["Missed"] == 5, (
        f"All 5 emergences should be missed, got {metrics['Missed']}"
    )
    print("  PASSED: lazy model correctly penalized\n")


def test_good_model_rewarded():
    """A model that detects emergence correctly should get MAE ≈ 0."""
    print("--- test_good_model_rewarded ---")
    num_pred = 12

    # Truth: emergence ramp
    truth = np.zeros((5, num_pred))
    for i in range(5):
        truth[i, 4:] = np.linspace(0, 0.5, num_pred - 4)

    # Good model: matches truth
    good_preds = truth.copy()

    metrics = compute_emergence_metrics(good_preds, truth)

    print(f"  Good MAE: {metrics['MAE']:.1f} (expected: 0)")
    print(f"  Detection Rate: {metrics['Detection_Rate']:.2f} (expected: 1.0)")

    assert metrics["MAE"] == 0.0, f"Good model should get MAE=0, got {metrics['MAE']}"
    assert metrics["Detection_Rate"] == 1.0, (
        f"Good model should have detection rate 1.0, got {metrics['Detection_Rate']}"
    )
    print("  PASSED: good model correctly rewarded\n")


def test_good_beats_lazy():
    """The good model must always score BETTER than the lazy model."""
    print("--- test_good_beats_lazy ---")
    num_pred = 12

    truth = np.zeros((5, num_pred))
    for i in range(5):
        truth[i, 4:] = np.linspace(0, 0.5, num_pred - 4)

    lazy_preds = np.ones((5, num_pred)) * 0.5
    good_preds = truth.copy()

    lazy_metrics = compute_emergence_metrics(lazy_preds, truth)
    good_metrics = compute_emergence_metrics(good_preds, truth)

    print(f"  Lazy MAE: {lazy_metrics['MAE']:.1f}")
    print(f"  Good MAE: {good_metrics['MAE']:.1f}")

    assert good_metrics["MAE"] < lazy_metrics["MAE"], (
        f"Good model MAE ({good_metrics['MAE']}) must be lower than lazy ({lazy_metrics['MAE']})"
    )
    print("  PASSED: good model beats lazy model\n")


def test_no_true_emergence():
    """When truth has no emergence, MAE=0 and detection rate=1.0 (nothing to miss)."""
    print("--- test_no_true_emergence ---")
    num_pred = 12

    # Both flat — no emergence in either
    truth = np.ones((3, num_pred)) * 0.5
    preds = np.ones((3, num_pred)) * 0.5

    metrics = compute_emergence_metrics(preds, truth)

    print(f"  MAE: {metrics['MAE']:.1f} (expected: 0)")
    print(f"  Detection Rate: {metrics['Detection_Rate']:.2f} (expected: 1.0)")

    assert metrics["MAE"] == 0.0, (
        f"No emergence should give MAE=0, got {metrics['MAE']}"
    )
    assert metrics["Detection_Rate"] == 1.0, (
        f"No emergence should give detection rate=1.0, got {metrics['Detection_Rate']}"
    )
    print("  PASSED: no-emergence case handled correctly\n")


def test_false_alarm_tracked():
    """When model predicts emergence but truth has none, false alarm is tracked."""
    print("--- test_false_alarm_tracked ---")
    num_pred = 12

    # Truth: flat
    truth = np.ones((3, num_pred)) * 0.5

    # Prediction: has emergence
    preds = np.zeros((3, num_pred))
    for i in range(3):
        preds[i, 4:] = np.linspace(0, 0.5, num_pred - 4)

    metrics = compute_emergence_metrics(preds, truth)

    print(f"  False Alarms: {metrics['False_Alarms']} (expected: 3)")
    print(f"  False Alarm Rate: {metrics['False_Alarm_Rate']:.2f}")

    assert metrics["False_Alarms"] == 3, (
        f"Expected 3 false alarms, got {metrics['False_Alarms']}"
    )
    assert metrics["False_Alarm_Rate"] == 1.0, (
        f"Expected 100% false alarm rate, got {metrics['False_Alarm_Rate']}"
    )
    print("  PASSED: false alarms correctly tracked\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Lazy Student Fix — Unit Tests")
    print("=" * 60)
    print()

    test_lazy_model_penalized()
    test_good_model_rewarded()
    test_good_beats_lazy()
    test_no_true_emergence()
    test_false_alarm_tracked()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
