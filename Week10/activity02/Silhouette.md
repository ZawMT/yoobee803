## Silhouette score
Silhouette score measures how well-separated and internally tight the clusters are — it's the standard way to judge K-Means quality without needing true labels (used in _print_cluster_report in clustering_analysis.py:30).

For each point it compares:
- a = average distance to other points in its own cluster (cohesion — want this small)
- b = average distance to points in the nearest other cluster (separation — want this large)
- score = (b - a) / max(a, b)

Averaged across all points, the result ranges from -1 to 1:
- Close to 1: tight, well-separated clusters
- Close to 0: clusters are overlapping/ambiguous
- Negative: points are probably in the wrong cluster (closer to a different cluster than their own)

In the actual results: k=2 scored 0.175, k=5 scored 0.113 — both fairly low, meaning the clusters K-Means found aren't strongly separated; there's a lot of overlap between groups in feature space. That's consistent with the ARI numbers from earlier (0.438 and 0.149) — the clinical features don't form crisp, naturally-separated clusters, they form a more continuous spread, which makes sense for medical risk data (risk is gradual, not categorical).

This is the "internal" metric — it's the one to actually trust in a real unsupervised setting where there is no result or target or labelled info (e.g. the patient is OK, Not OK, i.e. 'num' column) to check against, unlike ARI which only exists here because you happen to have ground truth for comparison.