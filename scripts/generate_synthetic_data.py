"""
Generates a synthetic 'transaction risk' dataset with a controllable drift
schedule, and pushes it into the feature store via the API.

Why synthetic + injected drift, instead of a random public dataset as-is:
you need KNOWN ground truth for when drift happens and how severe it is, so
you can honestly evaluate whether your PSI detector and retraining policies
catch it. This is standard practice in drift-detection research papers —
you'll cite this design choice explicitly in your writeup.

Usage:
    python scripts/generate_synthetic_data.py --hours 240 --events-per-hour 200
"""
import argparse
import time
import random
import requests
import numpy as np
from datetime import datetime, timedelta
API_URL = "http://localhost:8000"

FEATURE_NAME = "avg_transaction_amount"


def sample_feature_value(hour: int, drift_start_hour: int, drift_magnitude: float) -> float:
    """
    Baseline distribution: Normal(mean=50, std=10).
    After `drift_start_hour`, the mean gradually shifts upward by
    `drift_magnitude`, ramping in over 20 hours to simulate a realistic
    gradual shift rather than an instant jump (both are worth testing —
    try a hard jump too once the gradual case works).
    """
    base_mean = 50.0
    base_std = 10.0

    if hour >= drift_start_hour:
        ramp_hours = 20
        progress = min(1.0, (hour - drift_start_hour) / ramp_hours)
        mean = base_mean + drift_magnitude * progress
    else:
        mean = base_mean

    return max(0.0, np.random.normal(mean, base_std))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=int, default=240, help="Simulated hours of traffic to generate")
    parser.add_argument("--events-per-hour", type=int, default=200)
    parser.add_argument("--drift-start-hour", type=int, default=120, help="Hour at which drift begins")
    parser.add_argument("--drift-magnitude", type=float, default=25.0, help="How far the mean shifts")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between events (0 = as fast as possible)")
    args = parser.parse_args()

    print(f"Generating {args.hours}h of synthetic traffic, drift starting at hour {args.drift_start_hour}...")
    start_time = datetime.utcnow() - timedelta(hours=args.hours)

    for hour in range(args.hours):
        for _ in range(args.events_per_hour):
            value = sample_feature_value(hour, args.drift_start_hour, args.drift_magnitude)
            entity_id = f"user_{random.randint(1, 5000)}"
            try:
                requests.post(
                    f"{API_URL}/features/write",
                    json={"entity_id": entity_id, "feature_name": FEATURE_NAME, "value": value, "event_time": (start_time + timedelta(hours = hour, seconds=random.randint(0, 3599))).isoformat(),},
                    timeout=2,
                )
            except requests.RequestException as e:
                print(f"write failed: {e}")
            if args.sleep:
                time.sleep(args.sleep)

        if hour % 10 == 0:
            print(f"  hour {hour}/{args.hours} done")

    print("Done. Note: this script writes with event_time = now(), so for a realistic "
          "backfill of many simulated 'hours' in a short wall-clock run, you'll want to "
          "extend /features/write to accept an explicit event_time override for backfilled "
          "data. Left as your first real engineering task — see README Day 1-3.")


if __name__ == "__main__":
    main()
