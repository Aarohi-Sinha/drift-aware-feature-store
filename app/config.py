import os

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://fs_user:fs_pass@localhost:5432/feature_store")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")

# Drift + retraining policy knobs — tune these during your experiment phase.
PSI_WARNING_THRESHOLD = 0.1   # mild drift
PSI_ACTION_THRESHOLD = 0.25   # significant drift, worth acting on
RETRAIN_COMPUTE_COST_UNITS = 1.0   # placeholder "cost" of one retrain job, in arbitrary units
