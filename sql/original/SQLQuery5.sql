CREATE VIEW vw_ChurnedData AS
    SELECT * FROM prod_churn WHERE Customer_Status IN ('Churned', 'Stayed');
GO

CREATE VIEW vw_JoinedData AS
    SELECT * FROM prod_churn WHERE Customer_Status = 'Joined';
GO
