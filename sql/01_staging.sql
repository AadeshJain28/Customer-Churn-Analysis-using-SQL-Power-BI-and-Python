-- Staging load. DuckDB port of the original SQL Server script (sql/original/SQLQuery1.sql).
--
-- The original created [db_Churn].[dbo].[stg_churn] via the SSMS import wizard, which is
-- not reproducible from the repo. Here the raw CSV is the single source of truth and the
-- table is built by a statement anyone can re-run.

CREATE OR REPLACE TABLE stg_churn AS
SELECT * FROM read_csv_auto('data/raw/Customer_Data.csv', header = true);

-- Row-count assertion: the pipeline should refuse to continue on a truncated load.
-- DuckDB has no RAISERROR, so the check is expressed as a query the runner asserts on.
CREATE OR REPLACE VIEW stg_rowcount_check AS
SELECT
    COUNT(*)                        AS n_rows,
    COUNT(DISTINCT Customer_ID)     AS n_customers,
    COUNT(*) = COUNT(DISTINCT Customer_ID) AS customer_id_is_unique
FROM stg_churn;
