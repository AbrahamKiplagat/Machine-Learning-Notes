# 🌳 Decision Tree Classification – Beginner to Expert Guide

---

## 📌 What is a Decision Tree?

A **Decision Tree** is a **supervised learning algorithm** used for both **classification** and **regression** tasks. It works by recursively splitting the dataset into subsets based on **feature values** to form a tree-like model of decisions.

---

## 🧠 Intuition

- Each **internal node** represents a **decision** on a feature.
- Each **leaf node** represents a **class label** (in classification).
- The algorithm **splits** data at each node using a criterion to reduce impurity.

---

## 🧮 Splitting Criteria

1. **Gini Impurity**
   \[
   Gini = 1 - \sum_{i=1}^{C} p_i^2
   \]
   Lower is better. Default in `sklearn`.

2. **Entropy (Information Gain)**
   \[
   Entropy = - \sum_{i=1}^{C} p_i \log_2(p_i)
   \]
   Higher information gain = better split.

---

## 🛠 How the Tree is Built

1. Choose the best feature using a splitting criterion (e.g., Gini or Entropy).
2. Split the dataset based on the feature value.
3. Recur for each split until:
   - All data points belong to the same class, or
   - Maximum depth or minimum samples is reached.

---

## ✅ Pros

- Easy to understand and interpret.
- Handles both numerical and categorical data.
- Requires little data preprocessing (no scaling needed).

---

## ❌ Cons

- Prone to **overfitting**.
- Unstable (small changes in data → different tree).
- Biased towards features with more levels.

---

## 🧪 Practical Implementation (Sklearn)

### 1. Import Libraries
```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
