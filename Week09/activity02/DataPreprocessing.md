## User Knowledge Modeling — Data Preprocessing

| Step | Result |
| --- | --- |
| Null values | 0 found |
| Duplicate rows | 0 found |
| Label normalisation | Unified `very_low` → `Very Low` |
| Feature scaling | StandardScaler applied |
| Target encoding | LabelEncoder applied |

### Feature Scaling — StandardScaler

The 5 features (STG, SCG, STR, LPR, PEG) are measured on different scales. StandardScaler transforms each feature to have mean = 0 and standard deviation = 1. This prevents features with larger numeric ranges from dominating the model, and is especially important for KNN and Logistic Regression which are sensitive to scale.

### Target Encoding — LabelEncoder

The target column `UNS` contains text labels (`Very Low`, `Low`, `Middle`, `High`). Machine learning models require numbers, not text. LabelEncoder converts them to integers (e.g. `High` → 0, `Low` → 1, `Middle` → 2, `Very Low` → 3) based on alphabetical order. The model predicts these integers, and LabelEncoder converts them back to the original labels for reporting.