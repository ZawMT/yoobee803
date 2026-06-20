## ARI
ARI (Adjusted Rand Index) measures agreement between two different labelings — in this case, the K-Means cluster assignments vs. the real num diagnosis column(adjusted_rand_score(true_labels, cluster_labels) in clustering_analysis.py:32). It answers "do these clusters match the known answer?" — which requires having a known answer.

In a true unsupervised use case, for example, no diagnosis labels 'num' column, then ARI couldn't be computed. It is only possible to evaluate quality with silhouette/inertia alone.