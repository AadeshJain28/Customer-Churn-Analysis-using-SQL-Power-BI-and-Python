# What the model is actually worth

All figures from `reports/summary.json`, `baselines.csv`, `leaderboard.csv` and
`cost_sensitivity.csv`, on the same 1,202-customer held-out test set (347 churners).

## 1. The notebook reproduces

Before making any new claim, the pipeline reproduces the original notebook's model —
`RandomForestClassifier(n_estimators=100, random_state=42)` at threshold 0.5:

| Metric | Original notebook | This repo |
|---|---|---|
| Accuracy | 0.84 | **0.8478** |
| Precision (churn) | 0.78 | **0.8037** |
| Recall (churn) | 0.65 | **0.6254** |
| F1 (churn) | 0.71 | **0.7034** |

The rebuild is measuring the same thing the notebook measured. Everything below is a
change in *what is optimised*, not an artefact of a different pipeline.

The residual gap on recall (0.6254 vs 0.65) is an encoding difference, not noise: the
notebook used `LabelEncoder`, giving each Yes/No column a single ordinal 0/1 feature,
whereas this pipeline one-hot encodes them into a pair. For a tree the two are
equivalent in what they can express, but they change which split is chosen under a fixed
`random_state`, so the last decimal moves. Stating it beats implying the reproduction is
exact when it is close.

## 1a. The schema fix was behaviour-preserving

Between the first and second training runs, the numeric/categorical split moved from
being inferred at fit time to being declared in `config.yaml`, which re-routed 13 Yes/No
columns from `StandardScaler` to `OneHotEncoder` (see the train/serve skew section of
`reports/data_audit.md`).

A refactor that changes results is not a refactor. Every headline figure came back
identical:

| | Before the fix | After |
|---|---|---|
| Winner | gradient_boosting | gradient_boosting |
| Calibrated threshold | 0.13 | 0.13 |
| Recall | 0.9078 | **0.9078** |
| Precision | 0.5139 | **0.5139** |
| PR-AUC | 0.8260 | **0.8260** |
| Expected cost | 869.0 | **869.0** |
| ROC-AUC | 0.8958 | 0.8959 |

Only the notebook-reproduction row moved, for the encoding reason given above. This is
the "reduction to a known case" rung of the validation ladder used in reverse: a change
that should be invisible in the results was invisible in the results.

## 2. Expected cost, like for like

Cost = (customers contacted × 1) + (churners missed × 8).

| Strategy | Contacted | Missed | Cost | vs rule |
|---|---|---|---|---|
| Do nothing | 0 | 347 | 2,776.0 | — |
| Contact everyone | 1,202 | 0 | 1,202.0 | −25.8% |
| **`Contract == 'Month-to-Month'`** | 564 | 49 | **955.9** | — |
| **Gradient boosting, calibrated, t = 0.13** | 613 | 32 | **869.0** | **−9.1%** |

## 3. The honest reading

The model wins, and the margin is **9.1%** of retention spend.

It buys **+4.9 points of recall** (0.8588 → 0.9078) by contacting **49 more customers**
(46.9% → 51.0% of the base) to catch **17 more churners**. Precision is fractionally
*worse* than the rule's (0.5284 → 0.5139).

Stated plainly: a gradient-boosting model with calibrated probabilities and a
cost-optimised threshold beats one `if` statement by about 9% of budget. That is a real
improvement — 87 cost units per 1,202 customers, and it scales — but it is not the
order-of-magnitude gap the original 84%-accuracy headline implied, and anyone claiming the
model "identifies customers at risk of leaving" should know that contract type alone
identifies 86% of them.

P1 predicted no model would beat the rule's recall while flagging *fewer* customers. It
held: every model that beat 0.8588 recall did so by contacting a larger share of the base.
The gain is real but it is bought, not free.

## 4. The threshold is not stable — reported, not buried

`summary.json` records `threshold_stable_across_cost_ratios: false`. The sweep shows why:

| Assumed cost of a missed churner | Optimal threshold | Flagged | Recall |
|---|---|---|---|
| 2× | 0.51 | 21.9% | 0.6311 |
| 4× | 0.29 | 35.0% | 0.8012 |
| **6–8×** | **0.13** | **51.0%** | **0.9078** |
| 10× | 0.10 | 58.8% | 0.9395 |
| 15–20× | 0.03 | 77.1% | 0.9856 |

The operating point swings from contacting a fifth of the base to contacting three
quarters, depending on a cost ratio nobody has measured. The 8:1 figure in
`config/config.yaml` is an assumption, so **the threshold is only as defensible as that
assumption**. In deployment the ratio should be estimated from customer lifetime value and
campaign conversion, not inherited from a config file.

At 2× the model would flag 21.9% of customers — less than half the rule's reach — and the
rule would be the wrong policy for the opposite reason. The right conclusion is that the
cost ratio, not the algorithm, is the decision that matters most here.

## 5. Calibration behaved as predicted

Isotonic calibration moved ROC-AUC by −0.0011 (0.8969 → 0.8958) and improved the Brier
score (0.1086 → 0.1079), confirming P3: a monotone transform cannot reorder, so it changes
*where* the cost minimum sits without changing the ranking quality. The uncalibrated model
reaches a marginally lower raw cost (849.0 at t = 0.09) but its threshold is not
transferable, because the probability it corresponds to is not a probability.
