-- Distribution profiling. Port of sql/original/SQLQuery2.sql.
-- Each block became a named view so the Python layer and Power BI read the same
-- definition instead of re-implementing the arithmetic.

CREATE OR REPLACE VIEW profile_gender AS
SELECT Gender,
       COUNT(*)                                            AS total_count,
       COUNT(*) * 100.0 / (SELECT COUNT(*) FROM stg_churn) AS percentage
FROM stg_churn GROUP BY Gender ORDER BY percentage DESC;

CREATE OR REPLACE VIEW profile_contract AS
SELECT Contract,
       COUNT(*)                                            AS total_count,
       COUNT(*) * 100.0 / (SELECT COUNT(*) FROM stg_churn) AS percentage
FROM stg_churn GROUP BY Contract ORDER BY percentage DESC;

CREATE OR REPLACE VIEW profile_status_revenue AS
SELECT Customer_Status,
       COUNT(*)            AS total_count,
       SUM(Total_Revenue)  AS total_revenue,
       SUM(Total_Revenue) * 100.0 / (SELECT SUM(Total_Revenue) FROM stg_churn)
                           AS percentage_revenue
FROM stg_churn GROUP BY Customer_Status ORDER BY total_count DESC;

CREATE OR REPLACE VIEW profile_state AS
SELECT State,
       COUNT(*)                                            AS total_count,
       COUNT(*) * 100.0 / (SELECT COUNT(*) FROM stg_churn) AS percentage
FROM stg_churn GROUP BY State ORDER BY percentage DESC;

CREATE OR REPLACE VIEW profile_internet_type AS
SELECT Internet_Type,
       COUNT(*)                                            AS total_count,
       COUNT(*) * 100.0 / (SELECT COUNT(*) FROM stg_churn) AS percentage
FROM stg_churn GROUP BY Internet_Type ORDER BY percentage DESC;

-- Churn rate by contract. Not in the original script; added because it is the
-- single strongest driver in the data and it anchors the rule baseline in
-- reports/baselines.md.
CREATE OR REPLACE VIEW churn_rate_by_contract AS
SELECT Contract,
       COUNT(*)                                                   AS customers,
       SUM(CASE WHEN Customer_Status = 'Churned' THEN 1 ELSE 0 END) AS churned,
       SUM(CASE WHEN Customer_Status = 'Churned' THEN 1 ELSE 0 END) * 100.0 / COUNT(*)
                                                                  AS churn_rate_pct
FROM stg_churn
WHERE Customer_Status IN ('Churned', 'Stayed')
GROUP BY Contract ORDER BY churn_rate_pct DESC;
