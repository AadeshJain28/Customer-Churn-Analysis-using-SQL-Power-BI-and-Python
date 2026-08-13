# Power BI report

`Churn Analysis Project.pbix` is the descriptive dashboard from the original project:
demographics, contract and service mix, churn by category and reason, and revenue at risk.

It is kept as a binary artefact because Power BI Desktop is Windows-only and cannot be
built or tested in CI. That is a limitation, stated rather than hidden.

## Connecting it to this repo's data

The report originally pointed at a SQL Server instance (`db_Churn`). The equivalent
tables and views are now built by `make etl` into `data/processed/churn.duckdb`, using the
same definitions in `sql/`. To re-point the report:

1. Run `make etl`.
2. In Power BI Desktop: **Transform data → Data source settings**.
3. Point the source at the DuckDB file via the ODBC driver, or export the views first:

```bash
python -c "
from customer_churn.config import Config
from customer_churn.etl import connect
con = connect(Config.load())
for v in ['prod_churn','vw_churn_data','vw_join_data','vw_churn_reasons','churn_rate_by_contract']:
    con.execute(f\"COPY {v} TO 'data/processed/{v}.csv' (HEADER, DELIMITER ',')\")
"
```

## What the report should not be used for

`Churn_Category` and `Churn_Reason` are legitimate here — describing *why* customers left
is the report's purpose. They must not travel into the model: they are populated for 100%
of churners and 0% of stayers, so any classifier given them scores perfectly and learns
nothing. `sql/05_views.sql` excludes them from `vw_churn_data` for exactly this reason,
and `tests/test_leakage_guard.py` fails the build if that stops being true.
