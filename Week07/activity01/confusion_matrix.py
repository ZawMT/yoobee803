import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Construct labels matching the described outcomes:
# 15 actual Sick, 15 actual Healthy (equal split of 30 test records)
# FN = 2 (sick predicted as healthy), FP = 1 (healthy predicted as sick)
# TP = 13, TN = 14

y_true = ["Sick"] * 15 + ["Healthy"] * 15
y_pred = ["Sick"] * 13 + ["Healthy"] * 2 + ["Healthy"] * 14 + ["Sick"] * 1

labels = ["Sick", "Healthy"]
cm = confusion_matrix(y_true, y_pred, labels=labels)

print("Confusion Matrix:")
print(f"{'':>10} {'Pred Sick':>12} {'Pred Healthy':>14}")
print(f"{'Act Sick':>10} {cm[0][0]:>12} {cm[0][1]:>14}")
print(f"{'Act Healthy':>10} {cm[1][0]:>12} {cm[1][1]:>14}")
print()
print(f"TP (Sick correctly identified):    {cm[0][0]}")
print(f"FN (Sick missed, pred Healthy):    {cm[0][1]}")
print(f"FP (Healthy wrongly pred Sick):    {cm[1][0]}")
print(f"TN (Healthy correctly identified): {cm[1][1]}")
print()
print(classification_report(y_true, y_pred, labels=labels))

# Plot heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Pred: Sick", "Pred: Healthy"],
    yticklabels=["Act: Sick", "Act: Healthy"],
    linewidths=0.5,
    linecolor="gray",
)
plt.title("Confusion Matrix – Patient Health Classification", pad=14)
plt.ylabel("Actual Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
print("Figure saved as confusion_matrix.png")
