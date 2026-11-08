-- Offline feature store: append-only log of feature values, partition-friendly by day.
CREATE TABLE IF NOT EXISTS features (
    id BIGSERIAL PRIMARY KEY,
    entity_id TEXT NOT NULL,
    feature_name TEXT NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    event_time TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_features_entity ON features (entity_id, feature_name, event_time);
CREATE INDEX IF NOT EXISTS idx_features_time ON features (event_time);

-- Drift detection results, one row per (feature, evaluation window).
CREATE TABLE IF NOT EXISTS drift_events (
    id BIGSERIAL PRIMARY KEY,
    feature_name TEXT NOT NULL,
    window_start TIMESTAMPTZ NOT NULL,
    window_end TIMESTAMPTZ NOT NULL,
    psi_score DOUBLE PRECISION NOT NULL,
    triggered BOOLEAN NOT NULL DEFAULT false,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Every retraining decision + outcome, this is your headline research table.
CREATE TABLE IF NOT EXISTS retrain_events (
    id BIGSERIAL PRIMARY KEY,
    model_version TEXT NOT NULL,
    trigger_reason TEXT NOT NULL,       -- 'scheduled' | 'drift_threshold' | 'cost_aware' | 'manual'
    compute_cost_estimate DOUBLE PRECISION,
    accuracy_before DOUBLE PRECISION,
    accuracy_after DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
