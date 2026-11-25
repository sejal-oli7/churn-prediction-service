from contextlib import asynccontextmanager
from datetime import datetime, timezone

from bson import ObjectId
from fastapi import FastAPI, HTTPException, Query

from app.database import close_mongo_connection, connect_to_mongo, get_logs_collection
from app.log_schemas import LogEntry, LogListResponse
from app.ml_model import load_model, predict_churn
from app.schemas import CustomerFeatures, PredictionResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()                    # runs on startup
    await connect_to_mongo()
    yield
    await close_mongo_connection()  # runs on shutdown


app = FastAPI(title="Churn Prediction Service", lifespan=lifespan)


def serialize_log(doc: dict) -> dict:
    """MongoDB's _id is an ObjectId; JSON needs a string."""
    doc["id"] = str(doc.pop("_id"))
    return doc


def parse_object_id(log_id: str) -> ObjectId:
    """Turn the id from the URL into an ObjectId, or return a clean 400 error."""
    if not ObjectId.is_valid(log_id):
        raise HTTPException(status_code=400, detail="Invalid log_id format")
    return ObjectId(log_id)


@app.get("/health")
async def health():
    count = await get_logs_collection().count_documents({})
    return {"status": "ok", "logs_in_db": count}


@app.post("/predict", response_model=PredictionResponse)
async def predict(customer: CustomerFeatures):
    features = customer.model_dump()
    churn_prediction, confidence_score = predict_churn(features)

    log = {
        "input_features": features,
        "churn_prediction": churn_prediction,
        "confidence_score": confidence_score,
        "timestamp": datetime.now(timezone.utc),
    }
    result = await get_logs_collection().insert_one(log)

    return PredictionResponse(
        log_id=str(result.inserted_id),
        churn_prediction=churn_prediction,
        confidence_score=confidence_score,
    )


@app.get("/logs", response_model=LogListResponse)
async def list_logs(
    skip: int = Query(0, ge=0, description="How many logs to skip"),
    limit: int = Query(10, ge=1, le=100, description="Max logs to return"),
    churn_prediction: bool | None = Query(None, description="Filter: true or false"),
    min_confidence: float | None = Query(None, ge=0, le=1),
    max_confidence: float | None = Query(None, ge=0, le=1),
    start: datetime | None = Query(None, description="Logs from this time, e.g. 2026-11-25T00:00:00"),
    end: datetime | None = Query(None, description="Logs until this time"),
):
    # Build the MongoDB filter only from the options the user actually sent
    query: dict = {}
    if churn_prediction is not None:
        query["churn_prediction"] = churn_prediction

    if min_confidence is not None or max_confidence is not None:
        query["confidence_score"] = {}
        if min_confidence is not None:
            query["confidence_score"]["$gte"] = min_confidence
        if max_confidence is not None:
            query["confidence_score"]["$lte"] = max_confidence

    if start is not None or end is not None:
        query["timestamp"] = {}
        if start is not None:
            query["timestamp"]["$gte"] = start
        if end is not None:
            query["timestamp"]["$lte"] = end

    collection = get_logs_collection()
    total = await collection.count_documents(query)

    # newest first, then skip/limit for pagination
    cursor = collection.find(query).sort("timestamp", -1).skip(skip).limit(limit)
    docs = await cursor.to_list(length=limit)

    return LogListResponse(
        total=total, skip=skip, limit=limit, items=[serialize_log(d) for d in docs]
    )


@app.get("/logs/{log_id}", response_model=LogEntry)
async def get_log(log_id: str):
    doc = await get_logs_collection().find_one({"_id": parse_object_id(log_id)})
    if doc is None:
        raise HTTPException(status_code=404, detail="Log not found")
    return serialize_log(doc)


@app.delete("/logs/{log_id}")
async def delete_log(log_id: str):
    result = await get_logs_collection().delete_one({"_id": parse_object_id(log_id)})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Log not found")
    return {"message": "Log deleted", "deleted_id": log_id}
