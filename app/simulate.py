import pandas as pd
from xgboost import XGBClassifier

from app.config import PSI_ACTION_THRESHOLD, RETRAIN_COMPUTE_COST_UNITS
from app.db import SessionLocal, RetrainEvent
from app.drift import compute_psi
from app.model import load_data, TRAIN_HOURS
from app.policy import (
    PolicyDecision,
    fixed_schedule_policy,
    naive_threshold_policy,
    cost_aware_policy,
)

RETRAIN_WINDOW = 6          # retrain on the last 6 labeled hours
FIXED_EVERY = 6             # fixed schedule: every 6 hours
VALUE_PER_ACC_POINT = 5.0   # converts accuracy drop into the same units as cost
POLICIES = ["never", "fixed_schedule", "naive_threshold", "cost_aware"]

def fit(df: pd.DataFrame)->XGBClassifier:
    model = XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1)
    model.fit(df[["amount"]], df["label"])
    return model

def accuracy(model: XGBClassifier, df: pd.DataFrame) -> float:
    return float((model.predict(df[["amount"]]) == df["label"]).mean())


def run_policy(name: str, df: pd.DataFrame, last_hour: int):
    train = df[df["hour"] < TRAIN_HOURS]
    model = fit(train)
    baseline_amounts = train["amount"].values
    reference_acc = accuracy(model, train)

    retrains = 0
    accs = {}
    events = []

    for h in range(TRAIN_HOURS, last_hour + 1):
        prev = df[df["hour"] == h - 1]
        acc_prev = accuracy(model, prev)
        psi = compute_psi(baseline_amounts, prev["amount"].values)
        drop = max(0.0, reference_acc - acc_prev)
        i = h - TRAIN_HOURS

        if name == "never":
            decision = PolicyDecision(False, "never")
        elif name == "fixed_schedule":
            decision = fixed_schedule_policy(i + 1, FIXED_EVERY)
        elif name == "naive_threshold":
            decision = naive_threshold_policy(psi, PSI_ACTION_THRESHOLD)
        else:
            decision = cost_aware_policy(
                psi, PSI_ACTION_THRESHOLD, drop,
                RETRAIN_COMPUTE_COST_UNITS, VALUE_PER_ACC_POINT,
            )

        retrained = False
        if decision.should_retrain:
            window = df[(df["hour"] >= h - RETRAIN_WINDOW) & (df["hour"] < h)]
            model = fit(window)
            baseline_amounts = window["amount"].values
            reference_acc = accuracy(model, window)
            retrains += 1
            retrained = True

        acc_now = accuracy(model, df[df["hour"] == h])
        accs[h] = acc_now

        if retrained:
            events.append(RetrainEvent(
                model_version=f"{name}-h{h}",
                trigger_reason=decision.reason,
                compute_cost_estimate=RETRAIN_COMPUTE_COST_UNITS,
                accuracy_before=acc_prev,
                accuracy_after=acc_now,
            ))

    return retrains, accs, events


def main():
    df = load_data()
    last_hour = int(df["hour"].max())

    results = {}
    all_events = []
    for name in POLICIES:
        retrains, accs, events = run_policy(name, df, last_hour)
        results[name] = (retrains, accs)
        all_events.extend(events)

    with SessionLocal() as db:
        db.query(RetrainEvent).delete()
        db.add_all(all_events)
        db.commit()

    print("\n=== Summary (hours %d-%d) ===" % (TRAIN_HOURS, last_hour))
    print(f"{'policy':<16}{'retrains':>9}{'compute':>9}{'mean_acc':>10}{'worst_acc':>11}")
    for name in POLICIES:
        retrains, accs = results[name]
        vals = list(accs.values())
        print(f"{name:<16}{retrains:>9}{retrains * RETRAIN_COMPUTE_COST_UNITS:>9.1f}"
              f"{sum(vals) / len(vals):>10.3f}{min(vals):>11.3f}")

    print("\n=== Accuracy by hour ===")
    print(f"{'hour':>4}" + "".join(f"{n:>17}" for n in POLICIES))
    for h in range(TRAIN_HOURS, last_hour + 1, 3):
        print(f"{h:>4}" + "".join(f"{results[n][1][h]:>17.3f}" for n in POLICIES))


if __name__ == "__main__":
    main()