# Data audit — telecom customer churn

Every figure below is produced by `python -m customer_churn.audit`, which depends on
pandas alone and runs in CI on every push. Nothing here is typed by hand.

Source: `data/raw/Customer_Data.csv` — 6,418 customers, 32 columns.

| Split | Rows | Use |
|---|---|---|
| `Churned` + `Stayed` | **6,007** | Supervised training (`vw_churn_data`) |
| `Joined` | **411** | New customers, no outcome yet (`vw_join_data`) |

No duplicate `Customer_ID`s, no exact duplicate rows.

## 1. The churn rate is 28.8%, so accuracy is nearly uninformative

| Strategy | Accuracy | Precision | Recall | F1 | Share flagged |
|---|---|---|---|---|---|
| Predict "nobody churns" | **0.7117** | 0.0000 | 0.0000 | 0.0000 | 0% |
| Contact everybody | 0.2883 | 0.2883 | 1.0000 | 0.4476 | 100% |
| **`Contract == 'Month-to-Month'`** | 0.7348 | 0.5238 | **0.8828** | 0.6575 | 48.6% |
| `Tenure < 6 months` | 0.6396 | 0.2815 | 0.1611 | 0.2049 | 16.5% |

Reproduced by `make audit`; written to `reports/baselines.csv`.

The original notebook reported **84% accuracy** for a Random Forest. Against a constant
predictor at 71.2%, that is a **12.8-point** improvement, not an 84-point one. Any accuracy
figure quoted without the base rate beside it overstates the model by roughly that margin.

## 2. A one-line rule recovers 88% of churners

Flagging every month-to-month contract — no training, no features beyond one column —
catches **882 of every 1,000 churners** at 52.4% precision, contacting 48.6% of the base.

The notebook's Random Forest, at the default 0.5 threshold, reported **recall 0.65** on the
churn class (precision 0.78, F1 0.71). On recall — the quantity a retention team is
measured on — the if-statement beat the model by 23 points.

This is the central finding of the rebuild, and it is not an argument against modelling. It
reframes what the model has to do: **match the rule's reach while contacting far fewer
people**, which is a precision-and-threshold problem, not an accuracy problem. Hence
`evaluate.optimal_threshold`, which picks the operating point by expected cost rather than
leaving it at 0.5, and `reports/cost_sensitivity.csv`, which shows how far that point moves
as the assumed cost of a missed churner varies from 2× to 20× the cost of contact.

`tests/test_baselines.py` hand-derives all four baselines on a ten-row fixture and asserts
the real-data rule recall stays above 0.85, so a data refresh cannot quietly invalidate the
claim above.

## 3. `Churn_Category` and `Churn_Reason` are the label

| Column | Non-null when Churned | Non-null when Stayed |
|---|---|---|
| `Churn_Category` | **100%** | **0%** |
| `Churn_Reason` | **100%** | **0%** |

Either column alone separates the classes perfectly. A model given them reports flawless
metrics and has learned nothing. The original notebook drops them, which is correct; here
they are excluded in `sql/05_views.sql`, refused again by `features.split_xy`, and the
separation is re-measured by a test that fails the build if it changes.

They remain in `prod_churn` for the Power BI report, where describing *why* customers left
is the entire point and no prediction is being made. See `powerbi/README.md`.

## 4. Nulls are structural, not missing

| Column group | Null rows | Meaning |
|---|---|---|
| Internet add-on block (9 columns) | **1,223** | Customer has no internet service |
| `Value_Deal` | **3,297** | Customer is on no promotional deal |

The 1,223 rows are null across the whole add-on block simultaneously, not at random. The
SQL fills them with `'No'` / `'None'`, which encodes "does not have this feature" — the
truth. Leaving them null would let the encoder treat absence as its own category and learn
the service tier twice. `tests/test_sql.py::test_coalesce_removed_every_null_it_targeted`
asserts none survive the production build, and
`test_null_counts_agrees_with_pandas` cross-checks the SQL null audit against a pandas
implementation that shares no code with it — agreement is then evidence the port is
correct rather than evidence it is merely self-consistent.

## 5. Churners have the same tenure but half the revenue

| | Churned | Stayed |
|---|---|---|
| Mean tenure (months) | 17.52 | 17.35 |
| Mean total revenue | 1,969.95 | 3,745.06 |

Tenure is effectively identical (ratio 1.01) while revenue differs by a factor of 1.9, so
the gap is **not** an artefact of churned customers having been billed for less time — the
audit checks this explicitly and records `revenue_gap_explained_by_tenure: false`. Churners
genuinely spend around half as much per month. That is real signal, and it is worth stating
because the obvious reading — "of course churners billed less, they left" — is wrong here.

## Train/serve skew: the schema depended on the reader

The first version of `features.make_preprocessor` inferred the numeric/categorical split
at fit time with `select_dtypes(include=["number", "bool"])`. DuckDB's CSV sniffer types
the **13 Yes/No columns** as `BOOLEAN`, so they were routed to `StandardScaler` during
training. The dashboard built its input row from the raw CSV, where the same columns are
the strings `"Yes"`/`"No"`, and the transform raised
`ValueError: could not convert string to float: 'Yes'`.

| | StandardScaler | OneHotEncoder |
|---|---|---|
| Training frame (DuckDB) | 22 columns | 6 columns |
| Serving frame (raw CSV) | 9 columns | 19 columns |

The model was not wrong — a standardised boolean and a one-hot pair are equivalent to a
tree — but its idea of the schema depended on which reader loaded the data. **No amount of
cross-validation could have caught this**, because at training time nothing was
inconsistent. It is only visible from the serving side.

The fix: the split is declared in `config/config.yaml` instead of inferred, every frame
passes through `features.coerce_schema` at both ends, and the API and dashboard share
`prepare_inference_frame` so they cannot drift. `tests/test_schema.py` simulates DuckDB's
BOOLEAN typing and asserts a CSV-style and a DuckDB-style frame produce byte-identical
model matrices; CI runs it as its own step.

One further trap found while fixing it: the `null_fill` defaults must be **quoted** in
YAML. Unquoted, `No` parses as the boolean `false` under YAML 1.1, which would reintroduce
exactly the same bug through the config file. `Config.validate` now rejects that.

## What was wrong, and what caught it

| Claim | Status | Caught by |
|---|---|---|
| "Random Forest, 84% accuracy" | True but uninformative — the base rate is 71.2% | `baselines.py::majority_class` |
| Model is the right tool for the job | Partly false — a one-column rule gets 88% recall vs the model's 65% | `baselines.py::month_to_month_rule` |
| LabelEncoder before `train_test_split` | Fitted the category vocabulary on test rows | `features.make_preprocessor`, now fitted inside the pipeline |
| Threshold left at 0.5 | Not a decision, just a default | `evaluate.optimal_threshold` |
