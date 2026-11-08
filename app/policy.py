"""
Retraining trigger policies.

Three policies you'll compare head-to-head in your research experiment
(Part 8 of the project plan):

1. fixed_schedule   — retrain every N windows regardless of drift. Baseline.
2. naive_threshold  — retrain any time PSI crosses the action threshold. Common
                       in practice, but can over-trigger on noisy features.
3. cost_aware       — only retrain when predicted accuracy-recovery gain
                       exceeds the estimated compute cost. This is your novel
                       contribution: a simple expected-value calculation
                       instead of a bare threshold.

Keep this file pure/stateless so it's trivial to run in a simulation loop for
your experiment without spinning up the whole API.
"""
from dataclasses import dataclass


@dataclass
class PolicyDecision:
    should_retrain: bool
    reason: str


def fixed_schedule_policy(window_index: int, every_n_windows: int = 10) -> PolicyDecision:
    should = window_index % every_n_windows == 0
    return PolicyDecision(should, "scheduled" if should else "not_due")


def naive_threshold_policy(psi_score: float, action_threshold: float) -> PolicyDecision:
    should = psi_score >= action_threshold
    return PolicyDecision(should, "drift_threshold" if should else "below_threshold")


def cost_aware_policy(
    psi_score: float,
    action_threshold: float,
    estimated_accuracy_drop: float,
    retrain_cost_units: float,
    value_per_accuracy_point: float = 5.0,
) -> PolicyDecision:
    """
    A simple expected-value trigger: only retrain if the *value* of recovering
    the estimated accuracy drop outweighs the compute cost.

    estimated_accuracy_drop: your rolling estimate of how much downstream
        accuracy has degraded since the last retrain (you'll track this by
        periodically scoring the live model against recent labeled data).
    value_per_accuracy_point: an arbitrary unit conversion so cost and
        accuracy-gain are comparable — document your choice honestly in the
        writeup, this is exactly the kind of assumption reviewers will ask
        about, and having an explicit, named parameter here (rather than a
        hidden magic number) is what makes the policy defensible.
    """
    if psi_score < action_threshold:
        return PolicyDecision(False, "no_significant_drift")

    expected_value = estimated_accuracy_drop * value_per_accuracy_point
    should = expected_value > retrain_cost_units
    reason = "cost_aware_triggered" if should else "cost_aware_skipped_not_worth_it"
    return PolicyDecision(should, reason)
