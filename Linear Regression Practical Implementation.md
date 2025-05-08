
# 📊 Logistic Regression Algorithm

## ✅ Overview
Logistic Regression is a **supervised learning** algorithm used for **binary** and **multiclass classification** tasks. Unlike linear regression, it predicts **probability values** using a logistic (sigmoid) function.

---

## 📌 When to Use
- Predicting whether an email is spam or not
- Classifying customer churn (yes/no)
- Disease diagnosis (positive/negative)

---

## 🧠 Core Concept

### Sigmoid Function:
\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]
- Outputs values between **0 and 1**
- `z = w * x + b` (linear combination of weights and features)

---

## 🔧 Algorithm Steps

1. **Initialize** weights and bias
2. **Compute** linear combination: \( z = w \cdot x + b \)
3. **Apply sigmoid**: \( \hat{y} = \sigma(z) \)
4. **Compute loss** using Binary Cross-Entropy:
   \[
   L = -[y \log(\hat{y}) + (1 - y) \log(1 - \hat{y})]
   \]
5. **Update weights** using Gradient Descent
6. **Repeat** until convergence

---

## ⚙️ Example (Binary Classification)

```python
from sklearn.linear_model import LogisticRegression

# Training
model = LogisticRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)
```

---

## 📊 Output
- **Probability** of class membership
- Threshold (usually 0.5) used to classify:
  - ≥ 0.5 → Class 1
  - < 0.5 → Class 0

---

## 🚀 Advantages
- Simple and fast
- Interpretable coefficients
- Works well for linearly separable data

## ⚠️ Limitations
- Struggles with non-linear relationships
- Assumes no multicollinearity
- Sensitive to outliers

---

## 🔍 Variants
- **Multinomial Logistic Regression** (for >2 classes)
- **Regularized Logistic Regression** (L1/L2 penalties)

---


# 📈 Linear Regression - Practical Implementation Notes

## ✅ Overview
Linear Regression is a **supervised learning** algorithm used to model the relationship between a **dependent variable (y)** and one or more **independent variables (x)**.

---

## 🧠 Objective
To find the **best-fitting line**:
\[
y = w \cdot x + b
\]
Where:
- \( w \) is the **weight (slope)**
- \( b \) is the **bias (intercept)**

---

## 🛠️ Steps to Implement

### 1. Import Libraries
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
```

---

### 2. Load Dataset
```python
data = pd.read_csv('your_dataset.csv')  # Replace with your CSV path
print(data.head())
```

---

### 3. Preprocess Data
```python
X = data[['feature_column']]   # e.g., data[['YearsExperience']]
y = data['target_column']      # e.g., data['Salary']

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

### 4. Train Model
```python
model = LinearRegression()
model.fit(X_train, y_train)
```

---

### 5. Evaluate Model
```python
# Predict
y_pred = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse:.2f}')
print(f'R² Score: {r2:.2f}')
```

---

### 6. Visualize Results
```python
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Linear Regression Result')
plt.legend()
plt.show()
```

---

## 📊 Output
- Regression line
- Prediction metrics:
  - **MSE**: Lower is better
  - **R²**: Closer to 1 is better

---

## ✅ Advantages
- Easy to understand
- Quick to train
- Works well with linearly correlated data

## ⚠️ Limitations
- Assumes linear relationship
- Sensitive to outliers
- Can't capture complex patterns

---

## 🔁 Extension Ideas
- Try **Multiple Linear Regression** (use multiple features)
- Add **polynomial features** to capture non-linear trends
- Use **regularization** (Ridge, Lasso) to reduce overfitting

---

## 🧪 Sample Dataset Ideas
- Salary vs. Experience
- House Price vs. Area
- Revenue vs. Ad Spend

---

## 📁 References
- [Scikit-learn Linear Regression Docs](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- [Kaggle Datasets](https://www.kaggle.com/datasets)

---
