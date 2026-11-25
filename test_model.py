"""Check model + schemas (no MongoDB needed): python test_model.py"""
from app.ml_model import load_model, predict_churn
from app.schemas import CustomerFeatures

load_model()

customer = CustomerFeatures(
    gender="Male", seniorcitizen=0, partner="No", dependents="No",
    phoneservice="Yes", multiplelines="No", internetservice="Fiber optic",
    onlinesecurity="No", onlinebackup="No", deviceprotection="No",
    techsupport="No", streamingtv="No", streamingmovies="No",
    contract="Month-to-month", paperlessbilling="Yes",
    paymentmethod="Electronic check",
    tenure=2, monthlycharges=85.0, totalcharges=170.0,
)
print(customer.model_dump())
print(predict_churn(customer.model_dump()))

try:
    CustomerFeatures(**{**customer.model_dump(), "contract": "weekly"})
except Exception as e:
    print("Bad input rejected:", e.errors()[0]["msg"])
