Select Gender, Count(Gender) as TotalCount,
Count(Gender) * 100.0/ (Select Count(*) from stg_churn) as Percentage
From stg_churn
Group By Gender

Select Contract, Count(Contract) as TotalCount,
Count(Contract) * 100.0/ (Select Count(*) from stg_churn) as Percentage
From stg_churn
Group By Contract

Select Customer_Status, Count(Customer_Status) as TotalCount, Sum(Total_Revenue) as TotalRevenue,
Sum(Total_Revenue) / (Select Sum(Total_Revenue) from stg_churn) * 100 as PercentageRevenue
From stg_churn
Group By Customer_Status

Select State, Count(State) as TotalCount,
Count(State) * 100.0 / (Select Count(*) from stg_churn) as Percentage
from stg_churn
Group By State
Order By Percentage desc

Select Internet_Type, Count(Internet_Type) as TotalCount,
Count(Internet_Type) * 100.0/ (Select Count(*) from stg_churn) as Percentage
From stg_churn
Group By Internet_Type

Select Payment_Method, Count(Payment_Method) as TotalCount,
Count(Payment_Method) * 100.0/ (Select Count(*) from stg_churn) as Percentage
From stg_churn
Group By Payment_Method

Select Value_Deal, Count(Value_Deal) as TotalCount,
Count(Value_Deal) * 100.0/ (Select Count(*) from stg_churn) as Percentage
From stg_churn
Group By Value_Deal
Order By Percentage desc