# 📍 K-Nearest Neighbors (KNN) Algorithm – Intuition & Implementation

---

## 📌 What is KNN?

K-Nearest Neighbors (KNN) is a **non-parametric**, **lazy learning** algorithm used for both **classification** and **regression**.

- It **memorizes** the training dataset instead of learning a model.
- It **predicts** a new data point by looking at the “K” closest data points in the training set.

---

## 🔍 Intuition Behind KNN

Imagine a scatter plot of data points labeled by class (e.g., red vs. blue). To classify a new data point:
- Find the **K nearest neighbors** (using distance).
- Count the class labels among those K.
- Assign the **majority class** to the new point.

---

## 🧮 Distance Metrics

KNN relies on **distance** to determine "nearness". Common metrics:

- **Euclidean Distance**:
  \[
  d(p, q) = \sqrt{(p_1 - q_1)^2 + (p_2 - q_2)^2 + ... + (p_n - q_n)^2}
  \]

- **Manhattan Distance**:
  \[
  d(p, q) = |p_1 - q_1| + |p_2 - q_2| + ... + |p_n - q_n|
  \]

---

## ⚙️ Key Parameters

- **K (number of neighbors)**: odd values to avoid ties
- **Distance metric**: Euclidean is default
- **Weighting**: ‘uniform’ vs ‘distance’ (closer points weigh more)

---

## ✅ Pros
- Simple to implement
- No training required (lazy learner)
- Effective when decision boundary is nonlinear

---

## ❌ Cons
- **Slow at prediction** (computes distance from all points)
- **Sensitive to irrelevant features or outliers**
- Needs **feature scaling** (e.g., normalization)

---

## 🧪 Practical Implementation

### 1. Import Libraries
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```

## 2. Load Dataset
```python

data = pd.read_csv("your_dataset.csv")
X = data.drop("target", axis=1)
y = data["target"]
```
## 3. Preprocess (Scaling Important!)
```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```
## 4. Split the Data
```python
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
```
## 5. Train KNN Model
```python

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
```
## 6. Evaluate
```python

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\\n", confusion_matrix(y_test, y_pred))
```
## 🔧 Finding Best K (Hyperparameter Tuning)
```python
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

k_range = range(1, 21)
scores = [cross_val_score(KNeighborsClassifier(n_neighbors=k), X_scaled, y, cv=5).mean() for k in k_range]

plt.plot(k_range, scores)
plt.xlabel("K Value")
plt.ylabel("Cross-Validated Accuracy")
plt.title("K vs Accuracy")
plt.show()
```
## 🎯 Tips for Best Performance
Always scale your features before using KNN.

Try both uniform and distance weighting.

Start with odd values of K (e.g., 3, 5, 7).

Use GridSearchCV to find best K in a pipeline.

