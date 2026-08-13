"""Reference points the model must beat to have earned its existence.

This module is the reason the repo exists in its current form. The original notebook
reported 84% accuracy for a Random Forest and stopped. Two references it never computed:

  1. Predicting "nobody churns" scores 71.2% accuracy, because that is the base rate.
     The model's headline is therefore +12.8 points over a constant, not 84 points.

  2. A one-line rule -- flag every month-to-month contract -- recovers 88.3% of churners
     at 52.4% precision. The Random Forest at its default 0.5 threshold recovered 65%.
     On the metric a retention team actually cares about, the if-statement won.

Neither observation makes the model useless. Both change what can honestly be claimed
about it, and the second is the interesting result.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass
class BaselineScores:
    name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    flagged_share: float

    def as_dict(self) -> dict:
        return asdict(self)


def _score(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> BaselineScores:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return BaselineScores(
        name=name,
        accuracy=round(float((y_pred == y_true).mean()), 4),
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        flagged_share=round(float(y_pred.mean()), 4),
    )


def majority_class(y_true: np.ndarray) -> BaselineScores:
    """Predict the majority class for everyone. The floor for accuracy."""
    return _score("majority_class", y_true, np.zeros(len(y_true), dtype=int))


def flag_everyone(y_true: np.ndarray) -> BaselineScores:
    """Contact every customer. Recall 1.0 by construction; the precision ceiling
    of a no-model strategy, and the thing a cost analysis must beat."""
    return _score("flag_everyone", y_true, np.ones(len(y_true), dtype=int))


def month_to_month_rule(df: pd.DataFrame, y_true: np.ndarray) -> BaselineScores:
    """Flag every month-to-month contract. One column, no fitting, no training data."""
    return _score("rule_month_to_month", y_true, (df["Contract"] == "Month-to-Month").to_numpy())


def short_tenure_rule(df: pd.DataFrame, y_true: np.ndarray, months: int = 6) -> BaselineScores:
    """Flag customers under `months` tenure -- the other rule a manager would try."""
    return _score(f"rule_tenure_lt_{months}m", y_true, (df["Tenure_in_Months"] < months).to_numpy())


def all_baselines(df: pd.DataFrame, y_true: np.ndarray) -> pd.DataFrame:
    rows = [
        majority_class(y_true),
        flag_everyone(y_true),
        month_to_month_rule(df, y_true),
        short_tenure_rule(df, y_true),
    ]
    return pd.DataFrame([r.as_dict() for r in rows])
