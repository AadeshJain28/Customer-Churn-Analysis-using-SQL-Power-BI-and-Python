"""Metrics and the cost-optimal decision threshold."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass
class ClassifierScores:
    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    brier: float
    flagged_share: float
    expected_cost: float

    def as_dict(self) -> dict:
        return asdict(self)


def expected_cost(
    y_true: np.ndarray, y_pred: np.ndarray, intervention: float, missed: float
) -> float:
    """Cost of acting on these predictions.

    Contacting a customer costs `intervention` whether or not they would have churned.
    Missing a churner costs `missed`. True negatives cost nothing. This is the quantity
    a retention budget is actually spent against, and it is what the threshold optimises
    -- accuracy is not a business quantity.
    """
    y_true, y_pred = np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)
    contacted = int(y_pred.sum())
    missed_churners = int(((y_pred == 0) & (y_true == 1)).sum())
    return float(contacted * intervention + missed_churners * missed)


def score_at_threshold(
    y_true: np.ndarray, proba: np.ndarray, threshold: float, cfg
) -> ClassifierScores:
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_pred = (proba >= threshold).astype(int)
    c = cfg.raw["costs"]
    return ClassifierScores(
        threshold=round(float(threshold), 4),
        accuracy=round(float((y_pred == y_true).mean()), 4),
        precision=round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        recall=round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        f1=round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        roc_auc=round(float(roc_auc_score(y_true, proba)), 4),
        pr_auc=round(float(average_precision_score(y_true, proba)), 4),
        brier=round(float(brier_score_loss(y_true, proba)), 4),
        flagged_share=round(float(y_pred.mean()), 4),
        expected_cost=round(
            expected_cost(y_true, y_pred, c["intervention"], c["missed_churner"]), 2
        ),
    )


def optimal_threshold(y_true: np.ndarray, proba: np.ndarray, cfg) -> tuple[float, pd.DataFrame]:
    """Sweep thresholds and pick the one minimising expected cost.

    Returns the argmin and the full curve, so the choice can be inspected rather than
    trusted -- a flat minimum means the threshold is not really identified.
    """
    c = cfg.raw["costs"]
    grid = np.linspace(0.01, 0.99, 99)
    rows = []
    for t in grid:
        pred = (proba >= t).astype(int)
        rows.append(
            {
                "threshold": round(float(t), 3),
                "expected_cost": expected_cost(y_true, pred, c["intervention"], c["missed_churner"]),
                "flagged_share": float(pred.mean()),
                "recall": float(((pred == 1) & (y_true == 1)).sum() / max(y_true.sum(), 1)),
            }
        )
    curve = pd.DataFrame(rows)
    return float(curve.loc[curve.expected_cost.idxmin(), "threshold"]), curve


def cost_sensitivity(y_true: np.ndarray, proba: np.ndarray, cfg) -> pd.DataFrame:
    """How much does the chosen threshold depend on the assumed cost ratio?

    The 8:1 ratio in config is an estimate. If the optimal threshold swings wildly
    across plausible ratios, the recommendation is fragile and should be reported as such.
    """
    intervention = cfg.raw["costs"]["intervention"]
    rows = []
    for ratio in [2, 4, 6, 8, 10, 15, 20]:
        grid = np.linspace(0.01, 0.99, 99)
        costs = [
            expected_cost(y_true, (proba >= t).astype(int), intervention, intervention * ratio)
            for t in grid
        ]
        best = grid[int(np.argmin(costs))]
        pred = (proba >= best).astype(int)
        rows.append(
            {
                "cost_ratio": ratio,
                "optimal_threshold": round(float(best), 3),
                "flagged_share": round(float(pred.mean()), 4),
                "recall": round(float(((pred == 1) & (y_true == 1)).sum() / max(y_true.sum(), 1)), 4),
            }
        )
    return pd.DataFrame(rows)
