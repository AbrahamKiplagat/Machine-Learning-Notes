# Ensemble Learning: Bagging and Boosting
**Author**: Jonas Dieckmann  
**Summary by**: Abraham Kiplagat  
**Source**: TDS Archive on Medium  
**Date**: February 23, 2023

---

## 🔍 What Is Ensemble Learning?

Ensemble learning is a method in machine learning where **multiple models (learners)** are combined to solve a problem and improve performance.  
The core idea is that a **group of weak models** can come together to form a **strong learner**.

### 📌 Use Cases:
- Classification problems (e.g., image recognition)
- Regression analysis
- When facing **overfitting** or **underfitting**

---

## ⚖️ Why Use Ensemble Methods?

- **Single models** often have limited accuracy due to high **bias** or **variance**
- **Ensemble methods** aim to strike a **bias-variance tradeoff**
  - Reduce **variance** → Bagging
  - Reduce **bias** → Boosting

---

### Similarities
- Ensemble methods
In a general view, the similarities between both techniques start with the fact that both are ensemble methods with the aim to use multiple learners over a single model to achieve better results.
- Multiple samples & aggregation
To do that, both methods generate random samples and multiple training data sets. It is also similar that Bagging and Boosting both arrive at the end decision by aggregation of the underlying models: either by calculating average results or by taking a voting rank.
- Purpose
Finally, it is reasonable that both aim to produce higher stability and better prediction for the data.

## 🧠 Bias vs Variance Tradeoff

| Term | Description |
|------|-------------|
| **Bias** | Error from incorrect assumptions |
| **Variance** | Error from model sensitivity to small fluctuations |
| **Ideal Model** | Balances both, minimizing test error |

---

## 🧺 Bagging (Bootstrap Aggregation)

**Goal**: Reduce *variance* and avoid *overfitting*  
**How**: 
- Train multiple models **in parallel**
- Each model is trained on a **random subset (bag)** of data
- Final prediction: **Average (regression)** or **Majority vote (classification)**

### 🔧 Process:
1. Create multiple random samples from training data
2. Train the same base model (e.g., Logistic Regression) on each sample
3. Predict individually and aggregate results

### 📌 Characteristics:
- Reduces variance through averaging
- Models are **independent**
- Aggregation: simple average or vote
- Can be extended to **Random Forests** (also samples features)

---

## 🔍 Differences Between Bagging and Boosting

| Feature            | **Bagging**                                     | **Boosting**                                       |
|--------------------|--------------------------------------------------|----------------------------------------------------|
| **Goal**           | Reduce variance                                  | Reduce bias                                        |
| **Model Training** | Models trained independently in parallel         | Models trained sequentially                       |
| **Data Sampling**  | Random subsets with replacement                  | Weighted data – focuses on difficult samples       |
| **Model Weighting**| Equal weight for all models                      | Later models correct earlier errors with weights   |
| **Aggregation**    | Simple average or majority vote                  | Weighted sum of model predictions                  |
| **Overfitting Risk**| Lower (especially with decision trees)          | Higher, but can be controlled                      |
| **Examples**       | Random Forest                                    | AdaBoost, Gradient Boosting                        |

---
