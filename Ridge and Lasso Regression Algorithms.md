# Ridge and Lasso Regression Algorithms

---

## **Overview**
Ridge and Lasso Regression are **regularization techniques** used to prevent overfitting in linear regression models. They modify the loss function to penalize large coefficients, balancing model complexity and performance.

| **Aspect**          | **Ridge Regression (L2)**                 | **Lasso Regression (L1)**                 |  
|----------------------|-------------------------------------------|-------------------------------------------|  
| **Regularization**   | Adds penalty proportional to **squared coefficients**. | Adds penalty proportional to **absolute coefficients**. |  
| **Feature Selection**| Does not eliminate features (shrinks coefficients). | Can shrink coefficients to **zero** (selects features). |  
| **Use Case**         | Handles multicollinearity.                | Sparse models with fewer features.        |  

---

## **1. Ridge Regression (L2 Regularization)**
### **Definition**  
Ridge Regression adds a **penalty term** equal to the sum of the squares of the coefficients to the linear regression loss function.  
- **Goal**: Reduce model complexity by shrinking coefficients.  

### **Mathematical Formulation**  
\[
\text{Loss} = \text{MSE} + \lambda \sum_{i=1}^n \beta_i^2
\]  
- \(\lambda\) (alpha): Regularization strength (higher \(\lambda\) → stronger penalty).  
- \(\beta_i\): Coefficients of the model.  

### **Key Features**  
- **Handles Multicollinearity**: Stabilizes coefficient estimates when features are correlated.  
- **No Feature Elimination**: Coefficients approach zero but never exactly zero.  

### **Advantages**  
- Reduces overfitting.  
- Works well with correlated features.  

### **Disadvantages**  
- Does not perform feature selection.  

---

## **2. Lasso Regression (L1 Regularization)**  
### **Definition**  
Lasso Regression adds a **penalty term** equal to the sum of the absolute values of the coefficients.  
- **Goal**: Shrink coefficients and perform **feature selection**.  

### **Mathematical Formulation**  
\[
\text{Loss} = \text{MSE} + \lambda \sum_{i=1}^n |\beta_i|
\]  

### **Key Features**  
- **Feature Selection**: Forces some coefficients to zero, effectively removing irrelevant features.  
- **Sparsity**: Creates simpler, interpretable models.  

### **Advantages**  
- Automatically selects important features.  
- Useful for high-dimensional datasets.  

### **Disadvantages**  
- May arbitrarily select one feature from a group of correlated features.  

---

## **3. Key Differences**  
| **Aspect**          | **Ridge**                                | **Lasso**                                |  
|----------------------|------------------------------------------|------------------------------------------|  
| **Penalty Term**     | \(\lambda \sum \beta_i^2\) (L2 norm)     | \(\lambda \sum |\beta_i|\) (L1 norm)      |  
| **Coefficients**     | Shrinks but never zero.                  | Can shrink to zero.                      |  
| **Use Case**         | Correlated features, no feature selection. | Feature selection, sparse models.        |  
| **Interpretability** | Less interpretable (retains all features).| More interpretable (selects features).   |  

---

## **4. Practical Implementation in Python**  
```python
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Example: California Housing Dataset
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing()
X, y = data.data, data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features (critical for regularization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Ridge Regression
ridge = Ridge(alpha=1.0)  # λ = 1.0
ridge.fit(X_train_scaled, y_train)
y_pred_ridge = ridge.predict(X_test_scaled)
print(f"Ridge MSE: {mean_squared_error(y_test, y_pred_ridge):.2f}")

# Lasso Regression
lasso = Lasso(alpha=0.1)  # λ = 0.1
lasso.fit(X_train_scaled, y_train)
y_pred_lasso = lasso.predict(X_test_scaled)
print(f"Lasso MSE: {mean_squared_error(y_test, y_pred_lasso):.2f}")

# View coefficients (Lasso zeros out some features)
print("Lasso Coefficients:", lasso.coef_)