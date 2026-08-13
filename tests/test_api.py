from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.main import app

client = TestClient(app)
MODEL = Path(__file__).resolve().parents[1] / "models" / "churn_model.joblib"


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_rejects_invalid_contract():
    r = client.post("/predict", json={"Age": 40, "Tenure_in_Months": 5,
                                      "Contract": "Weekly", "Monthly_Charge": 70})
    assert r.status_code == 422


def test_rejects_impossible_age():
    r = client.post("/predict", json={"Age": 7, "Tenure_in_Months": 5,
                                      "Contract": "Month-to-Month", "Monthly_Charge": 70})
    assert r.status_code == 422


def test_empty_batch_rejected():
    assert client.post("/predict/batch", json=[]).status_code == 422


@pytest.mark.skipif(not MODEL.exists(), reason="model artefact absent; run `make train`")
def test_prediction_is_a_probability_and_reports_the_rule():
    r = client.post("/predict", json={"Age": 30, "Tenure_in_Months": 2,
                                      "Contract": "Month-to-Month", "Monthly_Charge": 90})
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["rule_baseline_would_flag"] is True
