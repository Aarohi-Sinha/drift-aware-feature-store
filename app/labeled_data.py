"""
Generates labeled transactions with drift and stores them in Postgres.

Label = 1 ("suspicious") when an amount is unusually high FOR THE CURRENT
REGIME (one std above that hour's mean). When the mean drifts upward,
the meaning of "unusually high" moves with it. 5% of labels are flipped
at random so accuracy is realistic instead of a perfect 100%.
"""

from datetime import datetime, timedelta, timezone

import numpy as np 
from sqlalchemy import text

from app.db import engine

START = datetime(2026, 11, 5, 1, 0, tzinfo=timezone.utc)
HOURS = 48
EVENTS_PER_HOUR = 200
DRIFT_START_HOUR = 24
RAMP_HOURS = 20
DRIFT_MAGNITUDE = 25.0
BASE_MEAN = 50.0
STD = 10.0
LABEL_NOISE = 0.05

def regime_mean(hour:int) -> float:
    if hour < DRIFT_START_HOUR:
        return BASE_MEAN
    progress = min(1.0, (hour - DRIFT_START_HOUR)/RAMP_HOURS)
    return BASE_MEAN + DRIFT_MAGNITUDE * progress

def main():
    rng = np.random.default_rng(42)
    rows = []
    for hour in range(HOURS):
        mean = regime_mean(hour)
        amounts = np.clip(rng.normal(mean, STD, EVENTS_PER_HOUR), 0, None)
        labels = (amounts> mean+STD).astype(int)
        flip = rng.random(EVENTS_PER_HOUR) < LABEL_NOISE
        labels= np.where(flip,1-labels, labels)
        for i in range(EVENTS_PER_HOUR):
            t = START + timedelta(hours = hour, seconds=int(rng.integers(0,3600)))
            rows.append({"t": t, "a": float(amounts[i]), "y":int(labels[i])})

    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS labeled_events(
                id BIGSERIAL PRIMARY KEY,
                event_time TIMESTAMPTZ NOT NULL,
                amount DOUBLE PRECISION NOT NULL,
                label INTEGER NOT NULL
            )
    """))
        conn.execute(text("TRUNCATE labeled_events"))
        conn.execute(
            text("INSERT INTO labeled_events(event_time, amount, label) VALUES (:t, :a, :y)"),
            rows,
        )
        print(f"Inserted {len(rows)} labeled events.")

if __name__ == "__main__":
    main()