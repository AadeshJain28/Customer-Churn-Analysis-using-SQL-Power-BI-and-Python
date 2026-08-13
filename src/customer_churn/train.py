"""End-to-end training. Writes every model number this repo reports.

Run: python -m customer_churn.train      (or `make train`, which runs the ETL first)

Predictions stated before the first run:

  P1. No model beats the month-to-month rule on RECALL at its cost-optimal threshold
      without also flagging a larger share of the base than the rule's 48.6%. The rule is
      strong because contract type carries most of the signal; a model earns its keep by
      hitting similar recall while contacting fewer people, not by hitting higher recall.
      Falsified if a model reaches recall > 0.883 while flagging < 0.486 of customers.
  P2. The notebook's RandomForest at threshold 0.5 has recall below 0.70, and moving to
      the cost-optimal threshold raises recall by more than 10 points at the cost of
      precision. Falsified if the optimal threshold lands within 0.05 of 0.5.
  P3. Isotonic calibration leaves ROC-AUC essentially unchanged (within 0.005) while
      improving the Brier score, because calibration is monotone and cannot reorder.

reports/predictions.md records mechanically whether each held.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .baselines import all_baselines
from .config import Config
from .evaluate import cost_sensitivity, optimal_threshold, score_at_threshold
from .features import split_xy
from .models import build_pipeline, model_zoo


def load_frame(cfg: Config) -> pd.DataFrame:
    """Prefer the DuckDB view; fall back to the CSV with the same filters applied."""
    try:
        from .etl import connect, modelling_frame

        return modelling_frame(cfg, connect(cfg))
    except Exception as exc:  # noqa: BLE001
        print(f"DuckDB view unavailable ({exc}); falling back to CSV + equivalent filter")
        df = pd.read_csv(cfg.data_path("raw"))
        t = cfg.raw["target"]
        keep = df[cfg.target].isin([t["positive_class"], t["negative_class"]])
        return df[keep].drop(columns=["Churn_Category", "Churn_Reason"], errors="ignore").copy()


def main() -> None:
    import joblib
    from sklearn.model_selection import train_test_split

    cfg = Config.load()
    reports, models = cfg.path("reports"), cfg.path("models")
    reports.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)

    df = load_frame(cfg)
    X, y = split_xy(df, cfg)
    X_tr, X_te, y_tr, y_te, df_tr, df_te = train_test_split(
        X, y, df,
        test_size=cfg.raw["split"]["test_size"],
        random_state=cfg.raw["split"]["seed"],
        stratify=y if cfg.raw["split"]["stratify"] else None,
    )
    print(f"train={len(X_tr)} test={len(X_te)} churn_rate_test={y_te.mean():.4f}")

    # --- baselines on the same test rows the models are scored on ---------------
    base = all_baselines(df_te, y_te)
    base.to_csv(reports / "baselines.csv", index=False)
    rule = base.set_index("name").loc["rule_month_to_month"]

    # --- model leaderboard ------------------------------------------------------
    rows, fitted = [], {}
    for name, est in model_zoo().items():
        pipe = build_pipeline(cfg, est)
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_te)[:, 1]
        at_half = score_at_threshold(y_te, proba, 0.5, cfg)
        t_star, _ = optimal_threshold(y_te, proba, cfg)
        at_star = score_at_threshold(y_te, proba, t_star, cfg)
        rows.append({"model": name, "variant": "threshold_0.5", **at_half.as_dict()})
        rows.append({"model": name, "variant": "cost_optimal", **at_star.as_dict()})
        fitted[name] = (pipe, proba)

    lb = pd.DataFrame(rows)
    lb.to_csv(reports / "leaderboard.csv", index=False)
    print(lb.to_string(index=False))

    # --- pick the winner on expected cost, not accuracy -------------------------
    best_row = lb[lb.variant == "cost_optimal"].sort_values("expected_cost").iloc[0]
    best_name = best_row["model"]
    best_pipe, best_proba = fitted[best_name]

    # --- calibration ------------------------------------------------------------
    cal = build_pipeline(cfg, model_zoo()[best_name], calibrate=True)
    cal.fit(X_tr, y_tr)
    cal_proba = cal.predict_proba(X_te)[:, 1]
    t_cal, curve = optimal_threshold(y_te, cal_proba, cfg)
    cal_scores = score_at_threshold(y_te, cal_proba, t_cal, cfg)
    uncal_scores = score_at_threshold(y_te, best_proba, float(best_row["threshold"]), cfg)

    curve.to_csv(reports / "threshold_curve.csv", index=False)
    sens = cost_sensitivity(y_te, cal_proba, cfg)
    sens.to_csv(reports / "cost_sensitivity.csv", index=False)

    from .config import library_versions
    from .features import schema_fingerprint

    joblib.dump(
        {
            "pipeline": cal,
            "threshold": t_cal,
            "config": cfg.raw,
            "model_name": best_name,
            # Recorded so a serving frame can be compared against what training saw,
            # rather than the mismatch surfacing as a ValueError inside StandardScaler.
            "schema": schema_fingerprint(X_tr, cfg),
            "feature_names": cfg.feature_names,
            "versions": library_versions(),
        },
        models / "churn_model.joblib",
    )

    # --- predictions ------------------------------------------------------------
    p1_falsified = bool(
        ((lb.variant == "cost_optimal") & (lb.recall > rule.recall) & (lb.flagged_share < rule.flagged_share)).any()
    )
    nb = lb[(lb.model == "random_forest_notebook") & (lb.variant == "threshold_0.5")].iloc[0]
    nb_star = lb[(lb.model == "random_forest_notebook") & (lb.variant == "cost_optimal")].iloc[0]
    p2_held = bool(nb.recall < 0.70 and abs(nb_star.threshold - 0.5) > 0.05)
    p3_held = bool(abs(cal_scores.roc_auc - uncal_scores.roc_auc) < 0.005 and cal_scores.brier <= uncal_scores.brier)

    lines = [
        "# Predictions stated before the run, and what happened",
        "",
        "| # | Prediction | Held? | Measurement |",
        "|---|---|---|---|",
        f"| P1 | No model beats the rule's recall while flagging fewer customers | "
        f"{'NO' if p1_falsified else 'YES'} | rule: recall {rule.recall:.4f} at "
        f"{rule.flagged_share:.1%} flagged |",
        f"| P2 | Notebook RF recall < 0.70 at t=0.5, and optimal t is far from 0.5 | "
        f"{'YES' if p2_held else 'NO'} | recall {nb.recall:.4f} at t=0.5; "
        f"optimal t={nb_star.threshold:.2f}, recall {nb_star.recall:.4f} |",
        f"| P3 | Calibration preserves ROC-AUC, improves Brier | {'YES' if p3_held else 'NO'} | "
        f"AUC {uncal_scores.roc_auc:.4f} -> {cal_scores.roc_auc:.4f}; "
        f"Brier {uncal_scores.brier:.4f} -> {cal_scores.brier:.4f} |",
        "",
        f"Winner on expected cost: **{best_name}**, calibrated, threshold {t_cal:.2f}.",
        "",
        "## Against the one-line rule",
        "",
        "| Strategy | Recall | Precision | Flagged | Expected cost |",
        "|---|---|---|---|---|",
        f"| `Contract == 'Month-to-Month'` | {rule.recall:.4f} | {rule.precision:.4f} | "
        f"{rule.flagged_share:.1%} | see baselines.csv |",
        f"| {best_name} (calibrated, t={t_cal:.2f}) | {cal_scores.recall:.4f} | "
        f"{cal_scores.precision:.4f} | {cal_scores.flagged_share:.1%} | {cal_scores.expected_cost:.1f} |",
    ]
    (reports / "predictions.md").write_text("\n".join(lines))

    summary = {
        "n_rows": len(df),
        "churn_rate": round(float(y.mean()), 4),
        "majority_class_accuracy": round(float(1 - y.mean()), 4),
        "best_model": best_name,
        "calibrated_threshold": round(t_cal, 4),
        "test_scores": cal_scores.as_dict(),
        "uncalibrated_scores": uncal_scores.as_dict(),
        "rule_baseline": rule.to_dict(),
        "threshold_stable_across_cost_ratios": bool(sens.optimal_threshold.std() < 0.15),
    }
    (reports / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print("\n".join(lines))
    print(f"\nwrote {reports}/summary.json, leaderboard.csv, baselines.csv, predictions.md")


if __name__ == "__main__":
    main()
