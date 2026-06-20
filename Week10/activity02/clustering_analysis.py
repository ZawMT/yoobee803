import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score


# ── Clustering Analysis (Unsupervised — 'num' withheld from training) ─────────

def _plot_clusters_pca(X_scaled, cluster_labels, true_labels, title, filename, output_dir):
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    var_ratio = pca.explained_variance_ratio_
    print(f"  PCA explained variance : PC1 {var_ratio[0]:.1%}, PC2 {var_ratio[1]:.1%} "
          f"(total {var_ratio.sum():.1%})")

    pc1_label = f"PC1 ({var_ratio[0]:.1%})"
    pc2_label = f"PC2 ({var_ratio[1]:.1%})"

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    sc0 = axes[0].scatter(coords[:, 0], coords[:, 1], c=cluster_labels, cmap="viridis", s=20)
    axes[0].set_title(f"{title} — K-Means Clusters")
    axes[0].set_xlabel(pc1_label)
    axes[0].set_ylabel(pc2_label)
    fig.colorbar(sc0, ax=axes[0])

    sc1 = axes[1].scatter(coords[:, 0], coords[:, 1], c=true_labels, cmap="viridis", s=20)
    axes[1].set_title(f"{title} — True Labels (for comparison only)")
    axes[1].set_xlabel(pc1_label)
    axes[1].set_ylabel(pc2_label)
    fig.colorbar(sc1, ax=axes[1])

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, filename), dpi=120)
    plt.close(fig)


def _print_cluster_report(name, X_scaled, cluster_labels, true_labels):
    print(f"\n--- {name} ---")
    print(f"  Silhouette Score                    : {silhouette_score(X_scaled, cluster_labels):.3f}")
    print(f"  Adjusted Rand Index vs. true labels : {adjusted_rand_score(true_labels, cluster_labels):.3f}")
    print("\n  Cluster vs. True Label cross-tab:")
    crosstab = pd.crosstab(pd.Series(cluster_labels, name="Cluster"),
                            pd.Series(true_labels.values, name="True Label"))
    print(crosstab.to_string())


def clustering_analysis(df, output_dir="charts/clustering"):
    print("\n" + "=" * 55)
    print("CLUSTERING ANALYSIS — K-Means (k=2 vs. binary disease label)")
    print("=" * 55)

    os.makedirs(output_dir, exist_ok=True)

    y = (df["num"] > 0).astype(int)
    X = df.drop(columns=["ID", "num"], errors="ignore")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    _print_cluster_report("K-Means (k=2)", X_scaled, cluster_labels, y)
    _plot_clusters_pca(X_scaled, cluster_labels, y, "K-Means (k=2)",
                        "kmeans_k2_pca.png", output_dir)

    print(f"\nCharts saved to '{output_dir}/'")


def clustering_analysis_multiclass(df, output_dir="charts/clustering/multiclass"):
    print("\n" + "=" * 55)
    print("CLUSTERING ANALYSIS — K-Means (k=5 vs. severity label 0–4)")
    print("=" * 55)

    os.makedirs(output_dir, exist_ok=True)

    y = df["num"]
    X = df.drop(columns=["ID", "num"], errors="ignore")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    _print_cluster_report("K-Means (k=5)", X_scaled, cluster_labels, y)
    _plot_clusters_pca(X_scaled, cluster_labels, y, "K-Means (k=5)",
                        "kmeans_k5_pca.png", output_dir)

    print(f"\nCharts saved to '{output_dir}/'")
