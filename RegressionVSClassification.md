# Regression vs Classification in Machine Learning

---

## **Regression**
### **Definition**  
- **Regression** is a supervised learning technique used to predict **continuous numerical values** (e.g., price, temperature, sales).  
- **Goal**: Model the relationship between dependent (target) and independent variables.  

### **Algorithms**  
1. Linear Regression  
2. Ridge/Lasso Regression  
3. Decision Tree Regression  
4. Random Forest Regressor  
5. XGBoost Regressor  

### **Key Concepts**  
- **Linear Regression Equation**:  
  \[
  Y = \beta_0 + \beta_1X_1 + \beta_2X_2 + \dots + \epsilon
  \]  
  - \(Y\): Target variable.  
  - \(\beta_0\): Intercept.  
  - \(\beta_1, \beta_2\): Coefficients.  
  - \(\epsilon\): Error term.  

- **Assumptions**:  
  - Linear relationship between features and target.  
  - Homoscedasticity (constant variance of errors).  
  - Normality of residuals.  

- **Evaluation Metrics**:  
  - **Mean Squared Error (MSE)**: \(\frac{1}{n}\sum (y_i - \hat{y}_i)^2\)  
  - **Mean Absolute Error (MAE)**: \(\frac{1}{n}\sum |y_i - \hat{y}_i|\)  
  - **R² (R-Squared)**: Proportion of variance explained by the model.  

- **Regularization**:  
  - **Ridge (L2)**: Penalizes large coefficients (\(\lambda \sum \beta_i^2\)).  
  - **Lasso (L1)**: Shrinks coefficients to zero for feature selection (\(\lambda \sum |\beta_i|\)).  

### **Use Cases**  
- Predicting house prices.  
- Forecasting stock market trends.  
- Estimating patient recovery time.  

### **Practical Tips**  
- Scale features (e.g., using `StandardScaler`).  
- Check for multicollinearity (e.g., VIF score).  
- Handle outliers (they heavily influence regression models).  

---

## **Classification**  
### **Definition**  
- **Classification** is a supervised learning technique used to predict **discrete labels** (e.g., spam/not spam, disease diagnosis).  
- **Goal**: Assign inputs to predefined categories.  

### **Algorithms**  
1. Logistic Regression  
2. Decision Tree Classifier  
3. Random Forest Classifier  
4. SVM (Support Vector Machines)  
5. K-Nearest Neighbors (KNN)  

### **Key Concepts**  
- **Logistic Regression**:  
  - Uses the **sigmoid function** to predict probabilities:  
    \[
    P(Y=1) = \frac{1}{1 + e^{-(\beta_0 + \beta_1X)}}
    \]  
  - Threshold (e.g., 0.5) converts probabilities to classes.  

- **Evaluation Metrics**:  
  - **Accuracy**: \(\frac{\text{Correct Predictions}}{\text{Total Predictions}}\).  
  - **Precision**: \(\frac{\text{TP}}{\text{TP + FP}}\) (minimize false positives).  
  - **Recall**: \(\frac{\text{TP}}{\text{TP + FN}}\) (minimize false negatives).  
  - **F1-Score**: Harmonic mean of precision and recall.  
  - **ROC-AUC**: Area under the ROC curve (trade-off between TPR and FPR).  

- **Handling Imbalanced Data**:  
  - Use **SMOTE** (oversampling) or **class weights**.  
  - Prefer **F1-Score** over accuracy for evaluation.  

### **Use Cases**  
- Email spam detection.  
- Medical diagnosis (e.g., cancer detection).  
- Customer churn prediction.  

### **Practical Tips**  
- Encode categorical features (e.g., one-hot encoding).  
- Tune hyperparameters (e.g., `max_depth` for Decision Trees).  
- Use **stratified sampling** for train-test splits in imbalanced datasets.  

---

## **Regression vs Classification: Comparison**  
| **Aspect**          | **Regression**                          | **Classification**                      |  
|----------------------|-----------------------------------------|-----------------------------------------|  
| **Output Type**      | Continuous (e.g., 10.5, 100.2)          | Discrete (e.g., 0/1, "spam"/"not spam") |  
| **Algorithms**       | Linear Regression, XGBoost Regressor    | Logistic Regression, SVM, KNN           |  
| **Evaluation**       | MSE, MAE, R²                            | Accuracy, Precision, Recall, ROC-AUC    |  
| **Loss Function**    | Mean Squared Error                      | Cross-Entropy Loss                       |  
| **Example Use Case** | Predicting sales revenue                | Detecting fraudulent transactions       |  

---

## **Summary**  
- **Regression** predicts **numbers**, while **Classification** predicts **labels**.  
- **Regression Metrics**: Focus on error magnitude (MSE, MAE).  
- **Classification Metrics**: Focus on class balance and correctness (Precision, Recall).  
- Choose algorithms based on problem type (e.g., Linear Regression for continuous outputs, Logistic Regression for binary labels).  