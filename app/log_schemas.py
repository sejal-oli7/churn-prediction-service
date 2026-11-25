from datetime import datetime

from pydantic import BaseModel


class LogEntry(BaseModel):
    """One document from the prediction_logs collection."""

    id: str
    input_features: dict
    churn_prediction: bool
    confidence_score: float
    timestamp: datetime


class LogListResponse(BaseModel):
    total: int      # how many logs match the filters
    skip: int
    limit: int
    items: list[LogEntry]
