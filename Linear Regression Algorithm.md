# Linear Regression Algorithm

---

## **Definition**  
Linear Regression is a **supervised learning algorithm** used to model the relationship between a **dependent variable (target)** and one or more **independent variables (features)** by fitting a linear equation to observed data.  

- **Goal**: Find the best-fit straight line that minimizes the prediction error.  
- **Type**: Regression (predicts continuous numerical values).  

---

## **Key Concepts**  
### 1. **Equation**  
- **Simple Linear Regression** (one feature):  
  \[
  Y = \beta_0 + \beta_1X + \epsilon
  \]  
- **Multiple Linear Regression** (multiple features):  
  \[
  Y = \beta_0 + \beta_1X_1 + \beta_2X_2 + \dots + \beta_nX_n + \epsilon
  \]  
  - \(Y\): Dependent variable (target).  
  - \(\beta_0\): Intercept (bias term).  
  - \(\beta_1, \beta_2, \dots, \beta_n\): Coefficients (weights) for features.  
  - \(\epsilon\): Error term (residuals).  

### 2. **Cost Function**  
- **Mean Squared Error (MSE)**:  
  \[
  \text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2
  \]  
  - \(y_i\): Actual value.  
  - \(\hat{y}_i\): Predicted value.  

### 3. **Optimization**  
- **Ordinary Least Squares (OLS)**:  
  Minimizes the MSE by finding optimal coefficients \(\beta_0, \beta_1, \dots, \beta_n\).  
- **Closed-Form Solution**:  
  \[
  \beta = (X^T X)^{-1} X^T y
  \]  
  (where \(X\) is the feature matrix and \(y\) is the target vector).  

---

## **Assumptions**  
1. **Linearity**: Relationship between features and target is linear.  
2. **Independence**: Residuals (errors) are uncorrelated.  
3. **Homoscedasticity**: Residuals have constant variance.  
4. **Normality**: Residuals are normally distributed.  
5. **No Multicollinearity**: Features are not highly correlated with each other.  

---

## **Evaluation Metrics**  
| Metric               | Formula                                  | Purpose                                |  
|----------------------|------------------------------------------|----------------------------------------|  
| **Mean Squared Error (MSE)** | \(\frac{1}{n}\sum (y_i - \hat{y}_i)^2\) | Penalizes larger errors.               |  
| **Mean Absolute Error (MAE)** | \(\frac{1}{n}\sum \|y_i - \hat{y}_i\|\) | Robust to outliers.                    |  
| **Root Mean Squared Error (RMSE)** | \(\sqrt{\text{MSE}}\)          | Interpretable in target units.         |  
| **R-Squared (\(R^2\))** | \(1 - \frac{\text{MSE}_{\text{model}}}{\text{MSE}_{\text{baseline}}}\) | Proportion of variance explained.      |  

---

## **Advantages & Disadvantages**  
| **Advantages**                     | **Disadvantages**                          |  
|------------------------------------|--------------------------------------------|  
| Simple and interpretable.          | Assumes linearity (fails on nonlinear data). |  
| Computationally efficient.         | Sensitive to outliers.                     |  
| Works well with small datasets.    | Prone to overfitting with many features.    |  

---

## **Practical Implementation in Python**  
```python
# Step 1: Import Libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing

# Step 2: Load Dataset
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='MedHouseVal')

# Step 3: Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Predict
y_pred = model.predict(X_test)

# Step 6: Evaluate
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.2f}")

# Step 7: Coefficients
print("Intercept (β₀):", model.intercept_)
print("Coefficients (β₁, β₂, ...):", model.coef_)