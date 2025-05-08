# Practical Implementation of Decision Tree Classifier

## 1. What is a Decision Tree?
A **Decision Tree** is a non-parametric machine learning algorithm used for **classification** and **regression** tasks. It builds models in the form of a tree structure by recursively splitting data into subsets based on feature values.  

### Key Concepts
- **CART (Classification and Regression Trees)**:  
  - Algorithm used in scikit-learn (`DecisionTreeClassifier`/`DecisionTreeRegressor`).  
  - Splits data hierarchically until stopping criteria (e.g., `max_depth`) are met.  
- **Structure**:  
  - **Root Node**: The top decision point.  
  - **Internal Nodes**: Splitting rules (e.g., "Petal Width ≤ 0.8").  
  - **Leaf Nodes**: Final predictions (class labels or continuous values).  
- **Advantages**:  
  - Easy to interpret and visualize.  
  - No need for feature scaling.  
  - Handles nonlinear relationships.  

---

## 2. How Decision Trees Work
### Classification
1. **Splitting Criteria**:  
   - **Gini Index**: Measures impurity (lower = better split).  
     \[
     \text{Gini} = 1 - \sum (p_i)^2
     \]  
   - **Entropy**: Measures disorder (lower = better split).  
     \[
     \text{Entropy} = -\sum p_i \log_2(p_i)
     \]  
2. **Prediction**:  
   - Traverse the tree from root to leaf based on feature values.  
   - Assign the **majority class** at the leaf node.  

### Regression
1. **Splitting Criteria**:  
   - Minimize **variance** in target values of child nodes.  
2. **Prediction**:  
   - Assign the **mean/median** of target values at the leaf node.  

---

### **1. Problem Definition**  
**Goal**: Classify data into discrete categories using a Decision Tree.  
**Example Use Case**: Classify iris flowers into species (setosa, versicolor, virginica) based on sepal/petal measurements.

---

### **2. Tools & Libraries**  
- **Python**  
- **Pandas**: Data handling.  
- **Scikit-learn**: Model training and evaluation.  
- **Matplotlib/Seaborn**: Visualization.  
- **Graphviz**: Tree visualization.  
