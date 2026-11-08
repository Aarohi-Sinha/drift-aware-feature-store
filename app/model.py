"""
Trains a baseline model on the FIRST 12 hours (before any drift), then scores it hour by hour on everything after, next to that hour's PSI
"""
import pandas as  pd
from sqlalchemy import text
from xgboost import XGBClassifier
from app.db import engine
from app.drift import compute_psi

TRAIN_HOURS = 12

def load_data() -> pd.DataFrame:
    df = pd.read_sql(
        text("SELECT event_time, amount, label FROM labeled_events ORDER BY event_time"),
        engine,
    )
    t0 = df["event_time"].min()
    df["hour"] = ((df["event_time"] - t0).dt.total_seconds() // 3600).astype(int)
    return df

def main():
    df = load_data()
    train = df[df["hour"] < TRAIN_HOURS]

    model = XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1)
    model.fit(train[["amount"]], train["label"])

    print(f"{'hour':>4} {'mean_amount':>12} {'accuracy':>9} {'psi_vs_train':>13}")
    for hour in range(TRAIN_HOURS, int(df["hour"].max()) + 1, 3):
        chunk = df[df["hour"] == hour]
        acc = (model.predict(chunk[["amount"]]) == chunk["label"]).mean()
        psi = compute_psi(train["amount"].values, chunk["amount"].values)
        print(f"{hour:>4} {chunk['amount'].mean():>12.1f} {acc:>9.3f} {psi:>13.3f}")


if __name__ == "__main__":
    main()