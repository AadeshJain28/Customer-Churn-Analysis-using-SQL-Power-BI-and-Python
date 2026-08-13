-- Production table. Port of sql/original/SQLQuery4.sql (SELECT ... INTO prod_churn).
-- SQL Server ISNULL(x, d) -> COALESCE(x, d), which is standard and portable.
--
-- The null-filling here is not cosmetic: 1,223 customers have NULL across the whole
-- internet-add-on block because they have no internet service. Filling with 'No' encodes
-- "does not have this feature", which is the truth; leaving NULL would let the encoder
-- treat absence as its own predictive category and quietly learn the service tier twice.

CREATE OR REPLACE TABLE prod_churn AS
SELECT
    Customer_ID,
    Gender,
    Age,
    Married,
    State,
    Number_of_Referrals,
    Tenure_in_Months,
    COALESCE(Value_Deal,             'None') AS Value_Deal,
    Phone_Service,
    COALESCE(Multiple_Lines,         'No')   AS Multiple_Lines,
    Internet_Service,
    COALESCE(Internet_Type,          'None') AS Internet_Type,
    COALESCE(Online_Security,        'No')   AS Online_Security,
    COALESCE(Online_Backup,          'No')   AS Online_Backup,
    COALESCE(Device_Protection_Plan, 'No')   AS Device_Protection_Plan,
    COALESCE(Premium_Support,        'No')   AS Premium_Support,
    COALESCE(Streaming_TV,           'No')   AS Streaming_TV,
    COALESCE(Streaming_Movies,       'No')   AS Streaming_Movies,
    COALESCE(Streaming_Music,        'No')   AS Streaming_Music,
    COALESCE(Unlimited_Data,         'No')   AS Unlimited_Data,
    Contract,
    Paperless_Billing,
    Payment_Method,
    Monthly_Charge,
    Total_Charges,
    Total_Refunds,
    Total_Extra_Data_Charges,
    Total_Long_Distance_Charges,
    Total_Revenue,
    Customer_Status,
    COALESCE(Churn_Category, 'Others')       AS Churn_Category,
    COALESCE(Churn_Reason,   'Others')       AS Churn_Reason
FROM stg_churn;
