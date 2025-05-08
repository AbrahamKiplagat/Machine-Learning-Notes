# 📊 Logistic Regression Algorithm

## 🔍 Overview
Logistic Regression is a **supervised learning** algorithm used for **binary classification** problems. Despite the name, it's actually a **classification** algorithm, not a regression one.

---

## 🧠 Key Concept

It models the **probability** that a given input belongs to a particular class using the **logistic (sigmoid) function**.

### Sigmoid Function:   σ(z) = 1 / (1 + e^(-z))
- Converts any real-valued number into a value between 0 and 1.
- Useful for predicting probabilities.

---

## ⚙️ Equation   z = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ
## p = σ(z) = 1 / (1 + e^(-z))


Where:
- `wᵢ` are weights
- `xᵢ` are input features
- `p` is the probability of class 1

---

## ✅ Classification Rule

- If `p ≥ 0.5`, predict **class 1**
- If `p < 0.5`, predict **class 0**

You can adjust the threshold based on the use case (e.g., 0.6 instead of 0.5).

---

## 📦 Cost Function

**Log Loss / Binary Cross-Entropy:**
##  J(θ) = -1/m ∑ [y log(h(x)) + (1 - y) log(1 - h(x))]


Used to measure the difference between actual and predicted values.

---

## 🛠️ Training

Uses **Gradient Descent** to minimize the cost function and find optimal weights `θ`.

---

## 🔍 Use Cases

- Spam detection
- Medical diagnosis (e.g., disease prediction)
- Customer churn prediction
- Credit scoring

---

## 🚧 Limitations

- Assumes linear decision boundary
- Sensitive to outliers
- Not suitable for non-linear problems unless transformed features are used

---

## 🔗 Extensions

- **Multinomial Logistic Regression**: for multi-class classification
- **Regularization (L1/L2)**: to avoid overfitting

---

## 🧪 Python Example (Scikit-learn)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy}")

