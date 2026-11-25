from typing import Literal

from pydantic import BaseModel, Field, field_validator

YesNo = Literal["yes", "no"]
YesNoNoInternet = Literal["yes", "no", "no_internet_service"]


class CustomerFeatures(BaseModel):
    """Input for /predict. Field names match the columns used in training."""

    gender: Literal["female", "male"]
    seniorcitizen: Literal[0, 1]
    partner: YesNo
    dependents: YesNo
    phoneservice: YesNo
    multiplelines: Literal["yes", "no", "no_phone_service"]
    internetservice: Literal["dsl", "fiber_optic", "no"]
    onlinesecurity: YesNoNoInternet
    onlinebackup: YesNoNoInternet
    deviceprotection: YesNoNoInternet
    techsupport: YesNoNoInternet
    streamingtv: YesNoNoInternet
    streamingmovies: YesNoNoInternet
    contract: Literal["month-to-month", "one_year", "two_year"]
    paperlessbilling: YesNo
    paymentmethod: Literal[
        "electronic_check",
        "mailed_check",
        "bank_transfer_(automatic)",
        "credit_card_(automatic)",
    ]
    tenure: int = Field(ge=0, le=100, description="Months with the company")
    monthlycharges: float = Field(ge=0)
    totalcharges: float = Field(ge=0)

    # Same cleaning as training: "Fiber optic" -> "fiber_optic", "Yes" -> "yes"
    @field_validator("*", mode="before")
    @classmethod
    def normalize_text(cls, v):
        if isinstance(v, str):
            return v.strip().lower().replace(" ", "_")
        return v


class PredictionResponse(BaseModel):
    log_id:str = Field(description="MongoDB _id of the saved log")
    churn_prediction: bool
    confidence_score: float = Field(description="Probability (0-1) that the customer churns")
