from __future__ import annotations

import pandas as pd
import pytest

from customer_churn.audit import load_raw, modelling_rows
from customer_churn.config import Config


@pytest.fixture(scope="session")
def cfg() -> Config:
    return Config.load()


@pytest.fixture(scope="session")
def raw(cfg) -> pd.DataFrame:
    return load_raw(cfg)


@pytest.fixture(scope="session")
def modelling(raw, cfg) -> pd.DataFrame:
    return modelling_rows(raw, cfg)


@pytest.fixture
def toy() -> pd.DataFrame:
    """Ten customers, hand-built so every baseline has a checkable answer.

    Churned: rows 0,1,2,3 (4 of 10) -> base rate 0.4, majority accuracy 0.6.
    Month-to-month: rows 0,1,2,5,6 (5 of 10).
    The rule therefore catches 3 of 4 churners (recall 0.75) with 3 of 5 correct
    (precision 0.6), and is right on 0,1,2 (TP) 3 (FN) 5,6 (FP) 4,7,8,9 (TN)
    -> accuracy 7/10 = 0.7. These are derived on paper, not from the code.
    """
    return pd.DataFrame(
        {
            "Contract": [
                "Month-to-Month", "Month-to-Month", "Month-to-Month", "Two Year",
                "One Year", "Month-to-Month", "Month-to-Month", "Two Year",
                "One Year", "Two Year",
            ],
            "Tenure_in_Months": [1, 3, 5, 40, 20, 30, 25, 60, 15, 50],
            "Customer_Status": [
                "Churned", "Churned", "Churned", "Churned", "Stayed",
                "Stayed", "Stayed", "Stayed", "Stayed", "Stayed",
            ],
        }
    )
