## Week 6 - Activity2: SVM Classification - wine dataset
Develop a script to train and test an SVM model, and share both the code and the results, including the evaluation metrics. Please refer to the provided dataset. link: https://archive.ics.uci.edu/dataset/109/wine

## Activity Note

The data is loaded and printed first 5 records.
![Loading data](./images/Explore1.png)

The data is explored and checked for any null.
![Data check](./images/Explore2.png)

The model is trained and evaluated — accuracy, confusion matrix, and classification report are printed.
![Training status](./images/TrainingStatus.png)

First, a 2D scatter plot on the features: Alcohol vs Flavanoids 
![Scatter plot 2D](./wine_outcome.png)

Second, a 3D scatter plot on the features: Flavanoids vs Alcohol vs Color intensity 
![Scatter plot 3D](./wine_outcome_3d.png)

Finally, a scatter plot showing the SVM decision boundary, projected to 2D using PCA.
![Scatter plot SVM](./wine_decision_boundary.png)
