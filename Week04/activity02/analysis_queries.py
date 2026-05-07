import duckdb
from tabulate import tabulate

DATA = "../data/world_happiness_dataset.csv"


def query_gdp_categories():
    print("\n--- Query 1: GDP Categories, Avg Happiness & Country Rankings ---\n")
    result = duckdb.query(f"""
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
    """).df()
    print(tabulate(result, headers="keys",
          tablefmt="rounded_outline", showindex=False))
    print("High GDP Category: GDP_per_Capita >= 1.3")
    print("Medium GDP Category: 0.9 <= GDP_per_Capita < 1.3")
    print("Low GDP Category: GDP_per_Capita < 0.9")


def query_corruption_comparison():
    print("\n--- Query 2: High vs Low Corruption — Average Comparisons ---\n")
    result = duckdb.query(f"""
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
    """).df()
    print(tabulate(result, headers="keys",
          tablefmt="rounded_outline", showindex=False))


def query_corruption_detail():
    print("\n--- Query 3: Each Country Sorted within Corruption Group ---\n")
    result = duckdb.query(f"""
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
        ORDER BY Corruption_Group, Rank_in_Group
    """).df()
    print(tabulate(result, headers="keys",
          tablefmt="rounded_outline", showindex=False))


def main():
    query_gdp_categories()
    query_corruption_comparison()
    query_corruption_detail()


if __name__ == "__main__":
    main()
