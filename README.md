# Churn Prediction Service

FastAPI service that predicts customer churn with a scikit-learn model
and logs every prediction to MongoDB.

## Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Dataset: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

## 1. Train the model
```bash
python train_model.py --data WA_Fn-UseC_-Telco-Customer-Churn.csv
```
Creates `churn_model.pkl` (DictVectorizer + LogisticRegression).

## 2. Start MongoDB
```bash
docker run -d --name mongo -p 27017:27017 -v mongo_data:/data/db mongo:7
```
Edit `.env` if you use a different MongoDB (e.g. Atlas).

## 3. Run the API
```bash
uvicorn app.main:app --reload
```
Docs: http://127.0.0.1:8000/docs

## Endpoints
| Method | Path | Description |
|---|---|---|
| POST | /predict | Predict churn and save the log |
| GET | /logs | List logs (skip, limit, churn_prediction, min_confidence, max_confidence, start, end) |
| GET | /logs/{log_id} | Get one log by MongoDB _id |
| DELETE | /logs/{log_id} | Delete one log |
| GET | /health | Check the service and MongoDB |

Example:
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" -d @customer.json
```

## Log document (collection `prediction_logs`)
`input_features`, `churn_prediction`, `confidence_score`, `timestamp`
