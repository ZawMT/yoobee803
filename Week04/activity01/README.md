## Week 4 Activity 1 : Data Visualization
Perform data visualization using the dataset provided in Week 4 on Blackboard. 
Before creating your visualizations, ensure that you conduct proper data cleaning, including handling missing values and detecting/removing outliers. Once your data is prepared, develop appropriate visualizations and provide a clear summary of your findings based on the visual analysis. Share your GitHub link when you have done.

### Data Exploration and Check
The given data is as follows:
![Given Data](/Week04/activity01/images/01-GivenData.png)

Checking if there is any missing value results as follows:
![Missing Value](/Week04/activity01/images/02-MissingValueCheck.png)

Calculating basic statistic values results as follows:
![Basic statistics](/Week04/activity01/images/03-BasicStatistics.png)

Checking duplicate data results as follows:
![Duplicate data](/Week04/activity01/images/04-DuplicateCheck.png)

Checking if there is any outliers using IQR method results as follows:
![Outliers](/Week04/activity01/images/05-OutlierCheck.png)

The IQR is calculated as 75% - 25%. Therefore, it is 6.26 - 4.24, so it is 2.02.
IQR method defines that any value below (Q1 - (1.5 × IQR)), or above (Q3 + (1.5 × IQR)) is flagged as an outlier.
Q1 is the lower quartile or 25% line of the data, and Q3 is upper quartile or 75%.

Checking data types results as follows:
![Data Types](/Week04/activity01/images/06-DataTypeCheck.png)

### Data Visualization
#### Horizontal bar chart
It clearly shows that, according to the given data, the happiest country is Canada, followed by Brazil and Finland. At the bottom, we can see Germany, Denmark and South Africa.

![Horizontal Bar Chart](/Week04/activity01/charts/happiness_bar.png)

#### Heatmap
It shows some interesting correlation between the features. 

There is anti-correlation between 'Freedom to Make Choices' and 'Generosity'. Probably, it might be that when a country is liberal, people tend to be more individualistic relying more strongly on personal choice and market systems, whereas when a country has less freedom, it have stronger community and collectivist culture, where helping others is a social norm.

The strongest correlation is between 'Social Support' vs 'Perception of Corruption'. It can be translated as countries with strong social support tend to have trusted institutions — and trust in institutions often goes hand-in-hand with lower perceived corruption. It means, alternatively, governments that invest in social support may also be more transparent and accountable.

![Heatmap](/Week04/activity01/charts/correlation_heatmap.png)

