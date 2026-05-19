import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

FEATURE_NAMES = [
    'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium',
    'Total phenols', 'Flavanoids', 'Nonflavanoid phenols',
    'Proanthocyanins', 'Color intensity', 'Hue',
    'OD280/OD315', 'Proline'
]
CLASS_NAMES = ['Class 1', 'Class 2', 'Class 3']


def load_data(filepath):
    df = pd.read_csv(filepath, header=None)
    df.columns = ['Class'] + FEATURE_NAMES
    return df


def explore_data(df):
    print("\nFirst 5 rows:")
    print(df.head())

    print("\nData types:")
    print(df.dtypes)

    print(f"\nDataset shape: {df.shape}")

    print(f"\nNull values:\n{df.isnull().sum()}")

    print(f"\nClass distribution:\n{df['Class'].value_counts().sort_index()}")

    print(f"\nBasic statistics:\n{df.describe()}")


def train_evaluate(df):
    X = df[FEATURE_NAMES].values
    y = df['Class'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42)

    model = SVC(kernel='linear')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(f"\nAccuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))


def plot_data(df):
    colors = ['red', 'green', 'blue']

    for i, cls in enumerate([1, 2, 3]):
        subset = df[df['Class'] == cls]
        plt.scatter(subset['Alcohol'], subset['Flavanoids'],
                    color=colors[i], label=CLASS_NAMES[i])

    plt.xlabel('Alcohol')
    plt.ylabel('Flavanoids')
    plt.title('Wine - Alcohol vs Flavanoids')
    plt.legend()
    plt.savefig('wine_outcome.png')
    plt.show()


def plot_data_3d(df):
    colors = ['red', 'green', 'blue']

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i, cls in enumerate([1, 2, 3]):
        subset = df[df['Class'] == cls]
        ax.scatter(subset['Flavanoids'], subset['Alcohol'],
                   subset['Color intensity'],
                   color=colors[i], label=CLASS_NAMES[i])

    ax.set_xlabel('Flavanoids')
    ax.set_ylabel('Alcohol')
    ax.set_zlabel('Color intensity')
    ax.set_title('Wine - Flavanoids vs Alcohol vs Color intensity')
    ax.view_init(elev=5, azim=50)
    ax.legend()
    plt.savefig('wine_outcome_3d.png')
    plt.show()


def plot_decision_boundary(df):
    X = df[FEATURE_NAMES].values
    y = df['Class'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Reduce to 2D with PCA so we can plot the boundary
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_scaled)

    # Train SVM on the 2D data
    model = SVC(kernel='linear')
    model.fit(X_2d, y)

    # Build a grid covering the 2D space
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))

    # Predict class for every point in the grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    colors = ['red', 'green', 'blue']
    bg_colors = ['#ffcccc', '#ccffcc', '#ccccff']

    plt.figure()
    plt.contourf(xx, yy, Z, levels=[0.5, 1.5, 2.5, 3.5],
                 colors=bg_colors, alpha=0.4)

    for i, cls in enumerate([1, 2, 3]):
        mask = y == cls
        plt.scatter(X_2d[mask, 0], X_2d[mask, 1],
                    color=colors[i], label=CLASS_NAMES[i], edgecolors='k', linewidths=0.4)

    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('SVM Decision Boundary (PCA 2D projection)')
    plt.legend()
    plt.savefig('wine_decision_boundary.png')
    plt.show()


def main():
    filepath = 'wine/wine.data'

    df = load_data(filepath)
    explore_data(df)
    train_evaluate(df)
    plot_data(df)
    plot_data_3d(df)
    plot_decision_boundary(df)


if __name__ == '__main__':
    main()
