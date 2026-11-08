from sqlalchemy import create_engine, Column, BigInteger, Text, Float, DateTime, Boolean, func
from sqlalchemy.orm import sessionmaker, declarative_base
import redis

from app.config import DATABASE_URL, REDIS_URL

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

redis_client = redis.from_url(REDIS_URL, decode_responses=True)


class FeatureRecord(Base):
    __tablename__ = "features"
    id = Column(BigInteger, primary_key=True)
    entity_id = Column(Text, nullable=False)
    feature_name = Column(Text, nullable=False)
    value = Column(Float, nullable=False)
    event_time = Column(DateTime(timezone=True), server_default=func.now())


class DriftEvent(Base):
    __tablename__ = "drift_events"
    id = Column(BigInteger, primary_key=True)
    feature_name = Column(Text, nullable=False)
    window_start = Column(DateTime(timezone=True), nullable=False)
    window_end = Column(DateTime(timezone=True), nullable=False)
    psi_score = Column(Float, nullable=False)
    triggered = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class RetrainEvent(Base):
    __tablename__ = "retrain_events"
    id = Column(BigInteger, primary_key=True)
    model_version = Column(Text, nullable=False)
    trigger_reason = Column(Text, nullable=False)
    compute_cost_estimate = Column(Float)
    accuracy_before = Column(Float)
    accuracy_after = Column(Float)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
