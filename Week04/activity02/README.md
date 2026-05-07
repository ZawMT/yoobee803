## Week 4 - Activity 2: Data Aggregation using W4-A1 dataset 
Develop two SQL quires based on the following two parts:

1. Creates GDP categories (Low, Medium, High), Calculates average happiness per category, Ranks countries within each category.

2. Splits countries into high vs low corruption perception, Computes multiple averages, Compares them using a subquery.

Explain your SQL queries in detail, including the purpose of each step and the logic behind your aggregations. Then, execute the queries and capture the final results as screenshots. Upload your work to GitHub and share the repository link.

### Query 1: Creates GDP categories and ranking
```sql
WITH categorised AS (
    SELECT
        Country,
        Happiness_Score,
        GDP_per_Capita,
        CASE
            WHEN GDP_per_Capita < 0.9  THEN 'Low'
            WHEN GDP_per_Capita < 1.3  THEN 'Medium'
            ELSE                            'High'
        END AS GDP_Category
    FROM read_csv_auto('{DATA}')
),
with_avg AS (
    SELECT
        Country,
        Happiness_Score,
        GDP_per_Capita,
        GDP_Category,
        ROUND(AVG(Happiness_Score) OVER (PARTITION BY GDP_Category), 2) AS Avg_Happiness_Category,
        RANK() OVER (PARTITION BY GDP_Category ORDER BY Happiness_Score DESC) AS Rank_in_Same_GDP_Category
    FROM categorised
)
SELECT *
FROM with_avg
ORDER BY
    CASE GDP_Category WHEN 'High' THEN 1 WHEN 'Medium' THEN 2 ELSE 3 END,
    Rank_in_Same_GDP_Category
```

​The query is using Common Table Expressions (CTEs). First CTE 'categorised' defines the categories of GDP_Per_Capita as Low, Medium and High. The second CTE 'with_avg' calculates the average of Happiness_Score, and defines the ranks per Happiness_Score in the same GDP_Category (as done in 'categorised'). Finally the data prepared in 'with_avg' is displayed, first the records with the value 'High' for 'GDP_Category', then followed by 'Medium', and then 'Low'.

![Result 1](/Week04/activity02/images/Table1.png)

### Query 2: Splits countries into high vs low corruption perception and computing averages
```sql
SELECT
    Corruption_Group,
    COUNT(*)                               AS Country_Count,
    ROUND(AVG(Happiness_Score), 2)         AS Avg_Happiness,
    ROUND(AVG(GDP_per_Capita), 2)          AS Avg_GDP,
    ROUND(AVG(Social_Support), 2)          AS Avg_Social_Support,
    ROUND(AVG(Healthy_Life_Expectancy), 2) AS Avg_Life_Expectancy,
    ROUND(AVG(Freedom_to_Make_Choices), 2) AS Avg_Freedom,
    ROUND(AVG(Generosity), 2)              AS Avg_Generosity
FROM (
    SELECT
        Country,
        Happiness_Score,
        GDP_per_Capita,
        Social_Support,
        Healthy_Life_Expectancy,
        Freedom_to_Make_Choices,
        Generosity,
        Perceptions_of_Corruption,
        CASE
            WHEN Perceptions_of_Corruption >= (
                SELECT AVG(Perceptions_of_Corruption)
                FROM read_csv_auto('{DATA}')
            )
            THEN 'High Corruption Perception'
            ELSE 'Low Corruption Perception'
        END AS Corruption_Group
    FROM read_csv_auto('{DATA}')
)
GROUP BY Corruption_Group
ORDER BY Corruption_Group
```

The subquery groups the data into two: 'High Corruption Perception' if Perceptions_of_Corruption is greater than equal to the average, and if not, 'Low Corruption Perception'. Then average values for other columns are calculated based on the two groups of data.

![Result 2](/Week04/activity02/images/Table2.png)

### Query 3: Splits countries into high vs low corruption perception and comparing 

```sql
SELECT
    Corruption_Group,
    Country,
    Happiness_Score,
    GDP_per_Capita,
    Perceptions_of_Corruption,
    RANK() OVER (PARTITION BY Corruption_Group ORDER BY Happiness_Score DESC) AS Rank_in_Group
FROM (
    SELECT
        Country,
        Happiness_Score,
        GDP_per_Capita,
        Perceptions_of_Corruption,
        CASE
            WHEN Perceptions_of_Corruption >= (
                SELECT AVG(Perceptions_of_Corruption)
                FROM read_csv_auto('{DATA}')
            )
            THEN 'High Corruption Perception'
            ELSE 'Low Corruption Perception'
        END AS Corruption_Group
    FROM read_csv_auto('{DATA}')
)
ORDER BY Corruption_Group, Rank_in_Group```

Similar to the previous query, but in this query, the data are sorted in the group that each belongs to.

![Result 3](/Week04/activity02/images/Table3.png)