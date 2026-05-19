### Notes
wine.names says that the classes are separable but only RDA has achieved 100% correct classification. The mentioned methods and terms are as follows:

#### Methods
**RDA** — Regularized Discriminant Analysis  
A flexible blend of LDA and QDA. It adds a "regularization" parameter to handle cases where one or the other breaks down. Most powerful of the three DA methods, hence
100% accuracy here.

**QDA** — Quadratic Discriminant Analysis    
Like LDA but allows each class to have its own shape/spread (covariance matrix). More flexible, but needs more data to estimate well. Got 99.4% here.

**LDA** — Linear Discriminant Analysis    
Finds a straight-line boundary to separate classes. Assumes all classes have the same spread. Simpler than QDA. Got 98.9% here.

**1NN** — 1-Nearest Neighbor   
Classifies a sample by finding the single most similar sample in the training data and copying its class label. Simple but sensitive to noise. Got 96.1% here (on
**standardized/z-transformed** data).

#### Terms
**Z-transformation** (also called standardization) rescales each feature so it has a mean of 0 and standard deviation of 1. It is a way of making features comparable to each other by removing the effect of different units and scales.
The formula for each value:
`z = (x - mean) / standard_deviation`
In this wine dataset, some features are on very different scales:
- Magnesium ranges ~70–162
- Malic acid ranges ~0.74–5.8
- Proline ranges ~278–1680
Without standardization, algorithms like 1NN get fooled — a small difference in Proline (large numbers) drowns out a large difference in Malic acid (small numbers), just because of the scale. Z-transformation puts all features on equal footing so no single feature dominates just because its numbers happen to be bigger. Technically it only needs to do this standardization for features with very different scales, but in practice it is done to all features — it's the standard convention — for the following reasons:
- Simpler and consistent — no need to manually judge which features need it
- Avoids accidentally missing a problematic feature
- Most ML pipelines just apply it to everything by default

Z-transformation only makes sense for numerical (continuous) features. The following features should be skipped:
- Class labels (categories like wine type 1/2/3)
- Categorical features (e.g. colour: "red", "white", "rosé")
- Binary features (e.g. 0/1 flags) — technically possible but rarely useful
For categorical features, a different encoding technique (like **one-hot encoding**) should be used before feeding them into a model.

**Leave-one-out technique** is a way of testing how accurate the model is without needing a separate test dataset.
In this wine dataset,  for example,
There are 178 wine samples. Testing will be repeated 178 times:
1. Take one sample out and set it aside
2. Train the model on the remaining 177 samples
3. Predict the class of the one held-out sample
4. Check if the prediction was correct
At the end, it will count how many of the 178 predictions were correct.

This technique is used for the following reasons:  
- Dataset is very small (in the current dataset, 178 samples is not a lot)
- To use as much data as possible for training
- It gives a reliable accuracy estimate without wasting data on a fixed test set
The tradeoff is that it is computationally expensive. In this small dataset, the training happens 178 times, but, on a large dataset (millions of rows) this would be too slow, so people use a simpler version called k-fold cross-validation instead. 

**PCA — Principal Component Analysis** is a dimensionality reduction technique that compresses many features into fewer summary dimensions while keeping as much information as possible. Each component is a weighted combination of all 13 original features. PCA finds the directions in the data where the most variation occurs and uses those as the new axes.

For example, Component 1 might end up being something like:
PC1 = 0.52 × Flavanoids + 0.47 × Total phenols + 0.40 × OD280/OD315 - 0.29 × Color intensity + ...
It's not any single original feature — it's a blend of all 13.

How does it pick the weights?
- Component 1 captures the direction of most variation in the data
- Component 2 captures the second most variation, at a right angle to Component 1
- And so on if more components are asked for.

As an analogy, imagine there is a 3D cloud of data points. If a light shines on it causing a shadow on the wall, the projection is changed from 3D to 2D. PCA finds the best angle to shine that light so the shadow preserves as much of the shape as possible.

It is noteworthy that PCA is a dimensionality reduction technique — it's a data transformation step, not a classification/learning method.
The distinction:
- Methods (algorithms that learn to make decisions): SVM, LDA, RDA, 1NN, etc.
- Dimensionality reduction (transforms data into fewer dimensions): PCA, t-SNE, UMAP, etc.
- Pre-processing (prepares data before training): StandardScaler, one-hot encoding, PCA, etc.
PCA sits in both the last two categories — it transforms the data and is used as a pre-processing step before the SVM in our case.

#### Related Terms
**K-fold cross-validation** is the same idea as leave-one-out, but instead of holding out 1 sample at a time, it holds out a chunk (fold) at a time. `k` refers to the number of rounds or folds in the following explanation. How it works (e.g. k=5): 
Split 178 samples into 5 equal chunks:
Fold 1: [1-36]    Fold 2: [37-71]    Fold 3: [72-107]    Fold 4: [108-142]    Fold 5: [143-178]
Then repeat 5 times:
┌───────┬───────────────┬─────────┐
│ Round │   Train on    │ Test on │
├───────┼───────────────┼─────────┤
│ 1     │ Folds 2,3,4,5 │ Fold 1  │
├───────┼───────────────┼─────────┤
│ 2     │ Folds 1,3,4,5 │ Fold 2  │
├───────┼───────────────┼─────────┤
│ 3     │ Folds 1,2,4,5 │ Fold 3  │
├───────┼───────────────┼─────────┤
│ 4     │ Folds 1,2,3,5 │ Fold 4  │
├───────┼───────────────┼─────────┤
│ 5     │ Folds 1,2,3,4 │ Fold 5  │
└───────┴───────────────┴─────────┘
Average the 5 accuracy scores → final accuracy estimate.
Common values of k:
- k=5 or k=10 are most common in practice
- k=178 (every sample) = leave-one-out

The tradeoff vs leave-one-out:
- Much faster (train 5 or 10 times instead of 178)
- Slightly less precise accuracy estimate, but good enough for most cases

**One-hot encoding** converts a categorical feature into multiple binary (0/1) columns — one column per category.
Example: A "Colour" feature with 3 categories:
┌────────┐
│ Colour │
├────────┤
│ Red    │
├────────┤
│ White  │
├────────┤
│ Rosé   │
├────────┤
│ Red    │
└────────┘
Becomes:
┌────────────┬──────────────┬─────────────┐
│ Colour_Red │ Colour_White │ Colour_Rosé │
├────────────┼──────────────┼─────────────┤
│ 1          │ 0            │ 0           │
├────────────┼──────────────┼─────────────┤
│ 0          │ 1            │ 0           │
├────────────┼──────────────┼─────────────┤
│ 0          │ 0            │ 1           │
├────────────┼──────────────┼─────────────┤
│ 1          │ 0            │ 0           │
└────────────┴──────────────┴─────────────┘
Each row has exactly one 1 and the rest 0 — hence "one-hot".

Why not just use numbers (1, 2, 3)?
If it is encoded like Red=1, White=2, Rosé=3, the model thinks Rosé is 3× bigger than Red, which is meaningless — there's no numeric relationship between wine colours. One-hot avoids this false ordering. 

The tradeoff:
If a feature has many categories (e.g. 100 countries), then it will result in 100 new columns — this can make the dataset very wide. This is called the **curse of dimensionality** and there are other encoding techniques to handle it, but one-hot is the simplest and most common.

In practice: one-hot encoding is the default starting point. It needs to reach for the others when there are too many categories or there is a specific reason. Just some quick info for some other methods.

**Label Encoding** simply assigns each category a number: Red=0, White=1, Rosé=2. Fast and simple, but implies a false order. Only suitable when the categories genuinely have an order (e.g. Small=0, Medium=1, Large=2) — those are called ordinal features.

**Target Encoding** replaces each category with the mean of the target variable for that category. For example, if wines labelled "Red" are class 1 on average 80% of the time, replace "Red" with 0.8. Compact but can cause data leakage if not done carefully.

**Frequency Encoding** replaces each category with how often it appears in the dataset. Red appears 50 times → replace with 50 (or 0.28 as a proportion). Simple, no extra columns.

**Binary Encoding** converts the category number to binary, then split each bit into a column. 100 countries → only ~7 columns instead of 100. A good middle ground between one-hot and label encoding.

**Hashing** maps categories into a fixed number of columns using a hash function. Very fast for very high-cardinality features but introduces some information loss.