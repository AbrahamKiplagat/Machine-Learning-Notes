# 🌳 Decision Tree Regression – Beginner to Expert Guide

---

## 📌 What is Decision Tree Regression?

**Decision Tree Regression** is a supervised machine learning algorithm used for **predicting continuous values** by learning **decision rules** from data features. It splits the data into regions and fits a constant value (average) in each region.

---

## 🔍 Intuition Behind It

- The algorithm **splits data recursively** based on input features to minimize prediction error.
- At each node, it finds the **best split** to minimize the **variance** of the target variable.
- The **prediction** for any region is simply the **mean of the target values** in that region.

---

## 🧮 Loss Function

- **Mean Squared Error (MSE)** is commonly used:
\[
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y})^2
\]

- The algorithm selects splits that reduce **total MSE** across resulting nodes.

---

## ✅ Pros

- Handles both numerical and categorical data.
- Easy to understand, visualize, and interpret.
- No need for feature scaling.
- Automatically captures non-linear relationships.

---

## ❌ Cons

- Can easily **overfit** (high variance).
- Not smooth (step-wise prediction).
- Sensitive to small changes in data.

---

## 🛠 Practical Implementation (with Scikit-learn)

### 1. Import Libraries
```python
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
