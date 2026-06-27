## User Knowledge Modeling — Data Analysis & Charts

### Clustering — K-Means

- Elbow method used to identify optimal k
- k=4 selected (matches number of knowledge level classes)
- PCA used for 2D visualisation of clusters vs actual labels

### Classification Results

| Model | Accuracy |
| --- | --- |
| Logistic Regression | 0.8642 |
| KNN | 0.8395 |
| Random Forest | 0.9259 |  **Best**
| XGBoost | 0.9136 |

**Best model: Random Forest**

### Charts

#### Class distribution
![01_class_balance.png](charts/01_class_balance.png)

#### Feature Distribution
![02_feature_distributions.png](charts/02_feature_distributions.png)

#### Boxplot by Class
![03_boxplots_by_class.png](charts/03_boxplots_by_class.png)

#### Heatmap
![04_correlation_heatmap.png](charts/04_correlation_heatmap.png)

#### Elbow Method (to choose K)
![05_elbow.png](charts/05_elbow.png)

#### Clustering with PCA
Clustering result is quite different from the actual labels, so it's more suitable to use supervised classification.
![06_clustering_pca.png](charts/06_clustering_pca.png)

#### Model Accuracy
![07_model_accuracy.png](charts/07_model_accuracy.png)

#### Confusion Matrix
![08_confusion_matrix.png](charts/08_confusion_matrix.png)

#### Feature Importance
![09_feature_importance.png](charts/09_feature_importance.png)
