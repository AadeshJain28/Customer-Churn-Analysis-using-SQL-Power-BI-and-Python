-- Modelling views. Port of sql/original/SQLQuery5.sql.
--
-- vw_churn_data is the supervised frame: only customers whose outcome is known.
-- vw_join_data is the scoring frame: new customers with no outcome yet.
--
-- Churn_Category and Churn_Reason are deliberately EXCLUDED from vw_churn_data. They are
-- populated for 100% of churned customers and 0% of stayed customers (see
-- reports/data_audit.md), so either column alone separates the classes perfectly. They
-- remain available in prod_churn for the Power BI report, where describing *why* people
-- left is the whole point and no prediction is being made.

CREATE OR REPLACE VIEW vw_churn_data AS
SELECT * EXCLUDE (Churn_Category, Churn_Reason)
FROM prod_churn
WHERE Customer_Status IN ('Churned', 'Stayed');

CREATE OR REPLACE VIEW vw_join_data AS
SELECT * EXCLUDE (Churn_Category, Churn_Reason)
FROM prod_churn
WHERE Customer_Status = 'Joined';

-- Kept separate and clearly named, for the descriptive report only.
CREATE OR REPLACE VIEW vw_churn_reasons AS
SELECT Churn_Category, Churn_Reason, COUNT(*) AS customers,
       COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () AS pct_of_churners
FROM prod_churn
WHERE Customer_Status = 'Churned'
GROUP BY Churn_Category, Churn_Reason
ORDER BY customers DESC;
