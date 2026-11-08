from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import select

from app.db import get_db, redis_client, FeatureRecord, DriftEvent, RetrainEvent
from app.drift import compute_psi, classify_psi
from app.policy import naive_threshold_policy, cost_aware_policy
from app.config import PSI_WARNING_THRESHOLD, PSI_ACTION_THRESHOLD, RETRAIN_COMPUTE_COST_UNITS
from app.dashboard import router as dashboard_router

app = FastAPI(title="Self-Optimizing Feature Store")
app.include_router(dashboard_router)

# ---------- Request/response schemas ----------

class FeatureWrite(BaseModel):
    entity_id: str
    feature_name: str
    value: float
    event_time: Optional[datetime] = None


class DriftCheckRequest(BaseModel):
    feature_name: str
    # Preferred: fixed windows, ISO timestamps like "2026-11-05T02:00:00Z"
    baseline_start: Optional[datetime] = None
    baseline_end: Optional[datetime] = None
    current_start: Optional[datetime] = None
    current_end: Optional[datetime] = None
     # Fallback: relative windows, used only if the fixed ones are not all given
    baseline_hours_ago_start: int = 48
    baseline_hours_ago_end: int = 24
    current_hours_ago_start: int = 24
    current_hours_ago_end: int = 0


# ---------- Feature write/read (the "feature store" core) ----------

@app.post("/features/write")
def write_feature(payload: FeatureWrite, db: Session = Depends(get_db)):
    # Offline store: durable, append-only, used for training + drift analysis.
    record = FeatureRecord(
        entity_id=payload.entity_id,
        feature_name=payload.feature_name,
        value=payload.value,
        event_time = payload.event_time or datetime.utcnow()
    )
    db.add(record)
    db.commit()

    # Online store: latest value only, fast lookups for live inference.
    redis_key = f"feature:{payload.entity_id}:{payload.feature_name}"
    redis_client.set(redis_key, payload.value)

    return {"status": "ok"}


@app.get("/features/{entity_id}/{feature_name}")
def read_feature(entity_id: str, feature_name: str):
    redis_key = f"feature:{entity_id}:{feature_name}"
    value = redis_client.get(redis_key)
    if value is None:
        return {"entity_id": entity_id, "feature_name": feature_name, "value": None, "source": "miss"}
    return {"entity_id": entity_id, "feature_name": feature_name, "value": float(value), "source": "online_store"}


# ---------- Drift detection ----------
def _as_utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)

@app.post("/drift/check")
def check_drift(req: DriftCheckRequest, db: Session = Depends(get_db)):
    if req.baseline_start and req.baseline_end and req.current_start and req.current_end:
        baseline_start = _as_utc(req.baseline_start)
        baseline_end = _as_utc(req.baseline_end)
        current_start = _as_utc(req.current_start)
        current_end = _as_utc(req.current_end)
    else:
        now = datetime.now(timezone.utc)
        baseline_start = now - timedelta(hours=req.baseline_hours_ago_start)
        baseline_end = now - timedelta(hours=req.baseline_hours_ago_end)
        current_start = now - timedelta(hours=req.current_hours_ago_start)
        current_end = now - timedelta(hours=req.current_hours_ago_end)

    baseline_vals = _fetch_values(db, req.feature_name, baseline_start, baseline_end)
    current_vals = _fetch_values(db, req.feature_name, current_start, current_end)

    if len(baseline_vals) < 30 or len(current_vals) < 30:
        return {"status": "insufficient_data", "baseline_n": len(baseline_vals), "current_n": len(current_vals)}

    psi = compute_psi(np.array(baseline_vals), np.array(current_vals))
    classification = classify_psi(psi, PSI_WARNING_THRESHOLD, PSI_ACTION_THRESHOLD)

    event = DriftEvent(
        feature_name=req.feature_name,
        window_start=current_start,
        window_end=current_end,
        psi_score=psi,
        triggered=classification == "significant_drift",
    )
    db.add(event)
    db.commit()

    return {
        "feature_name": req.feature_name,
        "psi_score": psi,
        "classification": classification,
        "baseline_n": len(baseline_vals),
        "current_n": len(current_vals),
        "baseline_mean": round(float(np.mean(baseline_vals)), 1),
        "current_mean": round(float(np.mean(current_vals)), 1),
        "baseline_window": [str(baseline_start), str(baseline_end)],
        "current_window": [str(current_start), str(current_end)],
    }


def _fetch_values(db: Session, feature_name: str, start: datetime, end: datetime):
    rows = db.execute(
        select(FeatureRecord.value)
        .where(FeatureRecord.feature_name == feature_name)
        .where(FeatureRecord.event_time >= start)
        .where(FeatureRecord.event_time < end)
    ).scalars().all()
    return list(rows)


@app.get("/drift/history")
def drift_history(db: Session = Depends(get_db), limit: int = 50):
    rows = db.execute(
        select(DriftEvent).order_by(DriftEvent.created_at.desc()).limit(limit)
    ).scalars().all()
    return [
        {
            "feature_name": r.feature_name,
            "psi_score": r.psi_score,
            "triggered": r.triggered,
            "window_start": r.window_start,
            "window_end": r.window_end,
        }
        for r in rows
    ]


# ---------- Retraining trigger + logging ----------

class RetrainDecisionRequest(BaseModel):
    psi_score: float
    estimated_accuracy_drop: float = 0.0
    policy: str = "cost_aware"  # 'naive_threshold' | 'cost_aware'


@app.post("/retrain/decide")
def decide_retrain(req: RetrainDecisionRequest):
    if req.policy == "naive_threshold":
        decision = naive_threshold_policy(req.psi_score, PSI_ACTION_THRESHOLD)
    else:
        decision = cost_aware_policy(
            req.psi_score,
            PSI_ACTION_THRESHOLD,
            req.estimated_accuracy_drop,
            RETRAIN_COMPUTE_COST_UNITS,
        )
    return {"should_retrain": decision.should_retrain, "reason": decision.reason}


@app.post("/retrain/log")
def log_retrain(
    model_version: str,
    trigger_reason: str,
    accuracy_before: Optional[float] = None,
    accuracy_after: Optional[float] = None,
    db: Session = Depends(get_db),
):
    event = RetrainEvent(
        model_version=model_version,
        trigger_reason=trigger_reason,
        compute_cost_estimate=RETRAIN_COMPUTE_COST_UNITS,
        accuracy_before=accuracy_before,
        accuracy_after=accuracy_after,
    )
    db.add(event)
    db.commit()
    return {"status": "logged"}


@app.get("/health")
def health():
    return {"status": "ok"}