"""FastAPI churn-scoring service.

    uvicorn app.api.main:app --reload

The service returns a probability, the decision at the cost-optimal threshold, and the
threshold itself -- so a caller can see *why* a customer was flagged and apply a different
operating point if their retention budget differs from the one in config.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "churn_model.joblib"
SUMMARY_PATH = ROOT / "reports" / "summary.json"

app = FastAPI(title="Customer Churn API", version="0.1.0")

YesNo = Literal["Yes", "No"]


class Customer(BaseModel):
    Gender: Literal["Male", "Female"] = "Male"
    Age: int = Field(..., ge=18, le=120)
    Married: YesNo = "No"
    State: str = "Uttar Pradesh"
    Number_of_Referrals: int = Field(0, ge=0, le=100)
    Tenure_in_Months: int = Field(..., ge=0, le=600)
    Value_Deal: str = "None"
    Phone_Service: YesNo = "Yes"
    Multiple_Lines: YesNo = "No"
    Internet_Service: YesNo = "Yes"
    Internet_Type: str = "Fiber Optic"
    Online_Security: YesNo = "No"
    Online_Backup: YesNo = "No"
    Device_Protection_Plan: YesNo = "No"
    Premium_Support: YesNo = "No"
    Streaming_TV: YesNo = "No"
    Streaming_Movies: YesNo = "No"
    Streaming_Music: YesNo = "No"
    Unlimited_Data: YesNo = "Yes"
    Contract: Literal["Month-to-Month", "One Year", "Two Year"]
    Paperless_Billing: YesNo = "Yes"
    Payment_Method: str = "Credit Card"
    Monthly_Charge: float = Field(..., ge=0, le=1000)
    Total_Charges: float = Field(0, ge=0)
    Total_Refunds: float = Field(0, ge=0)
    Total_Extra_Data_Charges: float = Field(0, ge=0)
    Total_Long_Distance_Charges: float = Field(0, ge=0)
    Total_Revenue: float = Field(0, ge=0)


class ChurnPrediction(BaseModel):
    churn_probability: float
    will_churn: bool
    threshold: float
    model_name: str
    rule_baseline_would_flag: bool


@lru_cache(maxsize=1)
def _bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"{MODEL_PATH} not found -- run `make train` first.")
    import joblib

    return joblib.load(MODEL_PATH)


def _to_model_frame(customers: list["Customer"]) -> pd.DataFrame:
    """Pydantic payload -> the exact schema the pipeline was fitted on.

    Shares `prepare_inference_frame` with the dashboard so the two cannot drift.
    """
    from customer_churn.config import Config
    from customer_churn.features import prepare_inference_frame

    raw = pd.DataFrame([c.model_dump() for c in customers])
    return prepare_inference_frame(raw, Config.load())


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_present": MODEL_PATH.exists()}


@app.get("/metadata")
def metadata() -> dict:
    if not SUMMARY_PATH.exists():
        raise HTTPException(404, "no summary.json -- run `make train`")
    return json.loads(SUMMARY_PATH.read_text())


@app.post("/predict", response_model=ChurnPrediction)
def predict(customer: Customer) -> ChurnPrediction:
    try:
        b = _bundle()
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc
    frame = _to_model_frame([customer])
    proba = float(b["pipeline"].predict_proba(frame)[0, 1])
    return ChurnPrediction(
        churn_probability=round(proba, 4),
        will_churn=proba >= b["threshold"],
        threshold=round(float(b["threshold"]), 4),
        model_name=b["model_name"],
        # Returned alongside the prediction so the caller can see when the model and the
        # trivial rule disagree -- the only cases where the model is adding anything.
        rule_baseline_would_flag=customer.Contract == "Month-to-Month",
    )


@app.post("/predict/batch")
def predict_batch(customers: list[Customer]) -> dict:
    if not customers:
        raise HTTPException(422, "empty batch")
    if len(customers) > 5000:
        raise HTTPException(413, "batch limit is 5000")
    try:
        b = _bundle()
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc
    frame = _to_model_frame(customers)
    proba = b["pipeline"].predict_proba(frame)[:, 1]
    flagged = (proba >= b["threshold"])
    return {
        "n": len(customers),
        "n_flagged": int(flagged.sum()),
        "threshold": round(float(b["threshold"]), 4),
        "churn_probability": [round(float(p), 4) for p in proba],
    }
