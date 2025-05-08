# 🤖 Naive Bayes Algorithm - Beginner to Expert Guide

---

## 🧠 What is Naive Bayes?

Naive Bayes is a **supervised learning** algorithm based on **Bayes’ Theorem** with an assumption of **independence among features**.

It is primarily used for:
- **Text classification**
- **Spam filtering**
- **Sentiment analysis**
- **Medical diagnosis**

---

## 📘 Bayes’ Theorem

\[
P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}
\]

Where:
- \( P(A|B) \): Posterior probability (A given B)
- \( P(B|A) \): Likelihood (B given A)
- \( P(A) \): Prior probability
- \( P(B) \): Evidence

---

## 🚨 Why “Naive”?

Because it **naively assumes** that features are **independent** of each other given the class, which rarely holds true in real-world data.

---

## 🛠 Types of Naive Bayes

| Type            | Description                           | Use Case             |
|-----------------|---------------------------------------|----------------------|
| **Gaussian**    | Assumes features follow a normal distribution | Continuous features  |
| **Multinomial** | Works with discrete counts            | Text classification  |
| **Bernoulli**   | Binary/boolean features               | Spam detection       |

---

## 🛠️ Steps to Implement Naive Bayes

### 1. Import Libraries
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
```

## 2. Load Dataset


```python
data = pd.read_csv("your_dataset.csv")
X = data.drop("target", axis=1)
y = data["target"]
```

## 3. Split the Data
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
## 4. Train Gaussian Naive Bayes
```python
model = GaussianNB()
model.fit(X_train, y_train)
```
## 5. Evaluate Model
```python
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\\n", classification_report(y_test, y_pred))
```
## 🧪 Text Classification with MultinomialNB
```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline

texts = ["I love this movie", "I hate this film", ...]
labels = [1, 0, ...]

X_train, X_test, y_train, y_test = train_test_split(texts, labels)

model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(X_train, y_train)
print("Score:", model.score(X_test, y_test))
```
## 🧮 Strengths of Naive Bayes
Fast and efficient

Performs well on high-dimensional data

Works well with small datasets

Effective for text classification

## ❌ Limitations
Assumption of feature independence is rarely true

Can struggle with correlated features

Not ideal for regression

## 🧙 Expert Tips
Use MultinomialNB for word count vectors

Apply Laplace smoothing to avoid zero probabilities

Normalize numeric features for GaussianNB

Use TF-IDF for better text vectorization

## 🔍 Tuning & Evaluation
```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(GaussianNB(), X, y, cv=5)
print("Cross-validated Accuracy:", np.mean(scores))
```