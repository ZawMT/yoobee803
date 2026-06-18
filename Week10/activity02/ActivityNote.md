## Week 10 - Activity 2 - Activity Notes

---

### Classification vs Clustering vs Prediction

#### What they have in common
- All three are core machine learning techniques
- All learn patterns from data to make decisions about new data
- All can be applied to the heart disease dataset

---

#### The Differences

| | **Classification** | **Clustering** | **Prediction (Regression)** |
|---|---|---|---|
| **Goal** | Assign data to a known category | Group similar data with no predefined labels | Estimate a continuous numeric value |
| **Output** | A label / class | A group / cluster number | A number |
| **Learning type** | Supervised (needs labelled data) | Unsupervised (no labels needed) | Supervised (needs labelled data) |
| **Example answer** | "This patient **has** heart disease" | "These patients form a **similar risk group**" | "This patient's salary is **$72,000**" |
| **Algorithms** | SVM, Decision Tree, Random Forest, KNN | K-Means, DBSCAN, Hierarchical | Linear Regression, Polynomial Regression |
| **Evaluation metric** | Accuracy, Precision, Recall, F1 | Silhouette score, Inertia | MAE, RMSE, R² |

---

#### In the context of Week 10 activities

| Activity | Technique | Why |
|---|---|---|
| `salary_prediction.py` | **Prediction** | Target is salary — a continuous number |
| `heart_disease.py` | **Classification** | Target is 0/1 — does the patient have heart disease or not |

Clustering would apply if you had the heart disease data but *no diagnosis labels* — you'd group patients by similarity and let the model discover patterns on its own.

---

#### Rule of Thumb

- Known labels + category output → **Classification**
- No labels, find natural groups → **Clustering**
- Known labels + number output → **Prediction (Regression)**
