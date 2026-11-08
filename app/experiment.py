"""
Phase 7 experiment: 10 random seeds x each policy x each setting of that
policy's one knob. Reports retrains (compute) and mean accuracy, averaged
over seeds, with the spread (std) across seeds.
"""
import numpy as np
import pandas as pd

from app import simulate
from app.labeled_data import regime_mean, HOURS, EVENTS_PER_HOUR, STD, LABEL_NOISE

SEEDS = range(10)
LAST_HOUR = HOURS - 1

DEFAULTS = {
    "FIXED_EVERY": simulate.FIXED_EVERY,
    "PSI_ACTION_THRESHOLD": simulate.PSI_ACTION_THRESHOLD,
    "VALUE_PER_ACC_POINT": simulate.VALUE_PER_ACC_POINT,
}

# (policy, knob name, knob value). Knobs are module settings in app.simulate.
CONFIGS = (
    [("never", None, None)]
    + [("fixed_schedule", "FIXED_EVERY", v) for v in (3, 6, 12)]
    + [("naive_threshold", "PSI_ACTION_THRESHOLD", v) for v in (0.1, 0.25, 1.0, 3.0)]
    + [("cost_aware", "VALUE_PER_ACC_POINT", v) for v in (2, 3, 5, 8, 12, 20)]
)


def make_df(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    for hour in range(HOURS):
        mean = regime_mean(hour)
        amounts = np.clip(rng.normal(mean, STD, EVENTS_PER_HOUR), 0, None)
        labels = (amounts > mean + STD).astype(int)
        flip = rng.random(EVENTS_PER_HOUR) < LABEL_NOISE
        labels = np.where(flip, 1 - labels, labels)
        parts.append(pd.DataFrame({"hour": hour, "amount": amounts, "label": labels}))
    return pd.concat(parts, ignore_index=True)


def reset_knobs():
    for k, v in DEFAULTS.items():
        setattr(simulate, k, v)


def main():
    datasets = {s: make_df(s) for s in SEEDS}
    print(f"{'policy':<16}{'knob':>22}{'retrains':>16}{'mean_acc':>18}")
    for policy, knob, value in CONFIGS:
        reset_knobs()
        if knob:
            setattr(simulate, knob, value)
        retrains, mean_accs = [], []
        for s in SEEDS:
            n, accs, _ = simulate.run_policy(policy, datasets[s], LAST_HOUR)
            retrains.append(n)
            mean_accs.append(sum(accs.values()) / len(accs))
        label = f"{knob}={value}" if knob else "-"
        print(f"{policy:<16}{label:>22}"
              f"{np.mean(retrains):>9.1f} ±{np.std(retrains):<4.1f}"
              f"{np.mean(mean_accs):>10.3f} ±{np.std(mean_accs):<6.3f}")
    reset_knobs()


if __name__ == "__main__":
    main()