import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Show first few records
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = [iris.target_names[i] for i in y]
print("\nFirst 5 records:")
print(df.head())

# Data quality check
df_check = pd.DataFrame(X, columns=iris.feature_names)
df_check['species'] = [iris.target_names[i] for i in y]
print("Data Types:")
print(df_check.dtypes)
print(f"\nNull values:\n{df_check.isnull().sum()}")
print(f"\nDataset shape: {df_check.shape}")

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train SVM classifier
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Scatter plot: sepal length vs petal length
colors = ['red', 'green', 'blue']
labels = iris.target_names

for i in range(3):
    plt.scatter(X[y == i, 0], X[y == i, 2], color=colors[i], label=labels[i])

plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.title('Iris - Sepal Length vs Petal Length')
plt.legend()
plt.savefig('test_outcome.png')
plt.show()
