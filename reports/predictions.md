# Predictions stated before the run, and what happened

| # | Prediction | Held? | Measurement |
|---|---|---|---|
| P1 | No model beats the rule's recall while flagging fewer customers | YES | rule: recall 0.8588 at 46.9% flagged |
| P2 | Notebook RF recall < 0.70 at t=0.5, and optimal t is far from 0.5 | YES | recall 0.6254 at t=0.5; optimal t=0.14, recall 0.9280 |
| P3 | Calibration preserves ROC-AUC, improves Brier | YES | AUC 0.8967 -> 0.8959; Brier 0.1087 -> 0.1079 |

Winner on expected cost: **gradient_boosting**, calibrated, threshold 0.13.

## Against the one-line rule

| Strategy | Recall | Precision | Flagged | Expected cost |
|---|---|---|---|---|
| `Contract == 'Month-to-Month'` | 0.8588 | 0.5284 | 46.9% | see baselines.csv |
| gradient_boosting (calibrated, t=0.13) | 0.9078 | 0.5139 | 51.0% | 869.0 |