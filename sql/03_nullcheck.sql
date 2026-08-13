-- Null audit. Port of sql/original/SQLQuery3.sql.
--
-- The original wrote 32 hand-rolled SUM(CASE WHEN col IS NULL THEN 1 ELSE 0 END)
-- expressions. This keeps one expression per column -- DuckDB's UNPIVOT needs a concrete
-- projection to pivot -- but transposes the wide row into (column_name, null_count) so the
-- result is a table you can filter and join rather than 32 columns to read sideways.
--
-- Drift is caught by tests/test_sql.py::test_null_counts_covers_every_column, which fails
-- if a column is added to the source and not listed here. That is a stronger guarantee
-- than the SQL alone: syntax cannot notice a missing column, a test can.
--
-- NOTE: the UNPIVOT target is `COLUMNS(*)`, with parentheses. Writing `ON COLUMNS` makes
-- DuckDB look for a column literally named COLUMNS and fail with a Binder Error.

CREATE OR REPLACE VIEW null_counts AS
UNPIVOT (
    SELECT
        COUNT(*) - COUNT(Customer_ID)                 AS Customer_ID,
        COUNT(*) - COUNT(Gender)                      AS Gender,
        COUNT(*) - COUNT(Age)                         AS Age,
        COUNT(*) - COUNT(Married)                     AS Married,
        COUNT(*) - COUNT(State)                       AS State,
        COUNT(*) - COUNT(Number_of_Referrals)         AS Number_of_Referrals,
        COUNT(*) - COUNT(Tenure_in_Months)            AS Tenure_in_Months,
        COUNT(*) - COUNT(Value_Deal)                  AS Value_Deal,
        COUNT(*) - COUNT(Phone_Service)               AS Phone_Service,
        COUNT(*) - COUNT(Multiple_Lines)              AS Multiple_Lines,
        COUNT(*) - COUNT(Internet_Service)            AS Internet_Service,
        COUNT(*) - COUNT(Internet_Type)               AS Internet_Type,
        COUNT(*) - COUNT(Online_Security)             AS Online_Security,
        COUNT(*) - COUNT(Online_Backup)               AS Online_Backup,
        COUNT(*) - COUNT(Device_Protection_Plan)      AS Device_Protection_Plan,
        COUNT(*) - COUNT(Premium_Support)             AS Premium_Support,
        COUNT(*) - COUNT(Streaming_TV)                AS Streaming_TV,
        COUNT(*) - COUNT(Streaming_Movies)            AS Streaming_Movies,
        COUNT(*) - COUNT(Streaming_Music)             AS Streaming_Music,
        COUNT(*) - COUNT(Unlimited_Data)              AS Unlimited_Data,
        COUNT(*) - COUNT(Contract)                    AS Contract,
        COUNT(*) - COUNT(Paperless_Billing)           AS Paperless_Billing,
        COUNT(*) - COUNT(Payment_Method)              AS Payment_Method,
        COUNT(*) - COUNT(Monthly_Charge)              AS Monthly_Charge,
        COUNT(*) - COUNT(Total_Charges)               AS Total_Charges,
        COUNT(*) - COUNT(Total_Refunds)               AS Total_Refunds,
        COUNT(*) - COUNT(Total_Extra_Data_Charges)    AS Total_Extra_Data_Charges,
        COUNT(*) - COUNT(Total_Long_Distance_Charges) AS Total_Long_Distance_Charges,
        COUNT(*) - COUNT(Total_Revenue)               AS Total_Revenue,
        COUNT(*) - COUNT(Customer_Status)             AS Customer_Status,
        COUNT(*) - COUNT(Churn_Category)              AS Churn_Category,
        COUNT(*) - COUNT(Churn_Reason)                AS Churn_Reason
    FROM stg_churn
)
ON COLUMNS(*)
INTO NAME column_name VALUE null_count;
