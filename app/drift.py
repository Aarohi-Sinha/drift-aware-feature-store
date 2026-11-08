"""
Drift detection using Population Stability Index (PSI).

PSI compares two distributions of the same feature (a "baseline" window,
usually from training time, and a "current" window, from live traffic) by
bucketing values and comparing bucket proportions.

Rule of thumb widely used in industry:
    PSI < 0.1  -> no significant change
    0.1 - 0.25 -> moderate change, worth watching
    > 0.25     -> significant change, distribution has meaningfully shifted

This file has zero framework dependencies on purpose — you can unit test it
with plain numpy arrays, which makes it easy to validate against your
synthetic drift-injection scripts.
"""
import numpy as np


def compute_psi(baseline: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """
    Compute PSI between two 1D arrays of numeric feature values.
    """
    baseline = np.asarray(baseline, dtype=float)
    current = np.asarray(current, dtype=float)

    # Build bin edges from the baseline distribution so "current" is measured
    # against the same reference frame it's being compared to.
    quantiles = np.linspace(0, 1, bins + 1)
    bin_edges = np.unique(np.quantile(baseline, quantiles))
    if len(bin_edges) < 3:
        # Degenerate case (baseline has almost no spread) — fall back to
        # simple linear bins across the combined range.
        lo = min(baseline.min(), current.min())
        hi = max(baseline.max(), current.max())
        bin_edges = np.linspace(lo, hi, bins + 1)

    baseline_counts, _ = np.histogram(baseline, bins=bin_edges)
    current_counts, _ = np.histogram(current, bins=bin_edges)

    # Convert to proportions, with a small epsilon to avoid log(0) / div-by-0
    eps = 1e-6
    baseline_props = baseline_counts / max(baseline_counts.sum(), 1) + eps
    current_props = current_counts / max(current_counts.sum(), 1) + eps

    psi = np.sum((current_props - baseline_props) * np.log(current_props / baseline_props))
    return float(psi)


def classify_psi(psi_score: float, warning_threshold: float, action_threshold: float) -> str:
    if psi_score >= action_threshold:
        return "significant_drift"
    if psi_score >= warning_threshold:
        return "moderate_drift"
    return "stable"
