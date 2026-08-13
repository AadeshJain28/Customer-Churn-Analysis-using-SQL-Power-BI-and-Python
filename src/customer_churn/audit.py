"""Data audit. Every number in reports/data_audit.md is produced here.

Deliberately depends on pandas only -- no scikit-learn, no DuckDB -- so the audit can be
reproduced in any environment, including CI, before any model exists.

Run: python -m customer_churn.audit
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .baselines import all_baselines
from .config import Config


@dataclass
class AuditResult:
    n_rows_raw: int
    n_rows_modelling: int
    n_rows_scoring: int
    duplicate_customer_ids: int
    exact_duplicate_rows: int
    churn_rate: float
    majority_class_accuracy: float
    churn_category_nonnull_when_churned: float
    churn_category_nonnull_when_stayed: float
    churn_reason_nonnull_when_churned: float
    churn_reason_nonnull_when_stayed: float
    internet_block_null_rows: int
    value_deal_null_rows: int
    mean_tenure_churned: float
    mean_tenure_stayed: float
    mean_revenue_churned: float
    mean_revenue_stayed: float
    revenue_gap_explained_by_tenure: bool


def load_raw(cfg: Config) -> pd.DataFrame:
    return pd.read_csv(cfg.data_path("raw"))


def modelling_rows(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    t = cfg.raw["target"]
    return df[df[cfg.target].isin([t["positive_class"], t["negative_class"]])].copy()


def run_audit(cfg: Config | None = None) -> tuple[AuditResult, pd.DataFrame]:
    cfg = cfg or Config.load()
    df = load_raw(cfg)
    m = modelling_rows(df, cfg)
    y = (m[cfg.target] == cfg.positive_class).astype(int).to_numpy()

    def nonnull_rate(frame: pd.DataFrame, col: str, status: str) -> float:
        sub = frame[frame[cfg.target] == status]
        return round(float(sub[col].notna().mean()), 4)

    # The internet add-on block is null as a unit for customers with no internet service.
    internet_cols = [
        "Internet_Type", "Online_Security", "Online_Backup", "Device_Protection_Plan",
        "Premium_Support", "Streaming_TV", "Streaming_Movies", "Streaming_Music", "Unlimited_Data",
    ]
    block_null = int(m[internet_cols].isna().all(axis=1).sum())

    ten_c, ten_s = m[y == 1].Tenure_in_Months.mean(), m[y == 0].Tenure_in_Months.mean()
    rev_c, rev_s = m[y == 1].Total_Revenue.mean(), m[y == 0].Total_Revenue.mean()
    # If tenure is equal but revenue differs, the gap is monthly spend, not billing history.
    tenure_ratio = ten_c / ten_s
    revenue_ratio = rev_c / rev_s
    explained = bool(abs(tenure_ratio - revenue_ratio) < 0.10)

    result = AuditResult(
        n_rows_raw=len(df),
        n_rows_modelling=len(m),
        n_rows_scoring=int((df[cfg.target] == "Joined").sum()),
        duplicate_customer_ids=int(df.Customer_ID.duplicated().sum()),
        exact_duplicate_rows=int(df.duplicated().sum()),
        churn_rate=round(float(y.mean()), 4),
        majority_class_accuracy=round(float(1 - y.mean()), 4),
        churn_category_nonnull_when_churned=nonnull_rate(m, "Churn_Category", "Churned"),
        churn_category_nonnull_when_stayed=nonnull_rate(m, "Churn_Category", "Stayed"),
        churn_reason_nonnull_when_churned=nonnull_rate(m, "Churn_Reason", "Churned"),
        churn_reason_nonnull_when_stayed=nonnull_rate(m, "Churn_Reason", "Stayed"),
        internet_block_null_rows=block_null,
        value_deal_null_rows=int(m.Value_Deal.isna().sum()),
        mean_tenure_churned=round(float(ten_c), 2),
        mean_tenure_stayed=round(float(ten_s), 2),
        mean_revenue_churned=round(float(rev_c), 2),
        mean_revenue_stayed=round(float(rev_s), 2),
        revenue_gap_explained_by_tenure=explained,
    )
    return result, all_baselines(m, y)


def main() -> None:
    cfg = Config.load()
    result, baselines = run_audit(cfg)
    reports = cfg.path("reports")
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "data_audit.json").write_text(json.dumps(asdict(result), indent=2))
    baselines.to_csv(reports / "baselines.csv", index=False)
    print(json.dumps(asdict(result), indent=2))
    print("\nBASELINES (no model involved)")
    print(baselines.to_string(index=False))
    print(f"\nwrote {reports}/data_audit.json and baselines.csv")


if __name__ == "__main__":
    main()
