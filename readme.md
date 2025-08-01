# Weather Prediction with Bagging and Random Forest - Solution Notebook

## Overview

This notebook (`5.desafio_solucion.ipynb`) provides a complete solution to the weather prediction challenge using ensemble methods. The goal is to predict whether it will rain tomorrow based on Australian weather data using **Bagging** and **Random Forest** techniques.

![Ensemble Methods Overview](./img/ensamble_01.png)

## Dataset

- **Source**: [Australian Weather Dataset - Kaggle](https://www.kaggle.com/jsphyg/weather-dataset-rattle-package)
- **Target Variable**: `RainTomorrow` (Yes/No - binary classification)
- **Features**: Various weather measurements including temperature, humidity, pressure, wind speed, etc.

## Notebook Structure

### 1. Exploratory Data Analysis (EDA) and Preprocessing

#### Data Loading and Cleaning

```python
data = pd.read_csv("../Data/weatherAUS.csv.zip")
```

The preprocessing steps include:

- **Removing columns with insufficient data** (< 100,000 non-null values)
- **Dropping location and date columns** for generalization
- **Eliminating categorical variables** to simplify preprocessing
- **Removing rows with null values**

#### Feature Selection Strategy

Based on correlation analysis, highly correlated features are identified and removed:

- `Temp3pm` and `Pressure9am` are dropped due to high correlation with other features

#### Target Variable Preparation

```python
data['RainTomorrow'] = data['RainTomorrow'].map({'Yes':1,'No':0})
```

#### Baseline Models

Two benchmark models are created:

- **Always "No Rain"**: Predicts 0 for all instances
- **Always "Rain"**: Predicts 1 for all instances

These provide baseline accuracy scores to compare ensemble performance against.

### 2. Manual Bagging Implementation

![Bootstrap Sampling](./img/bootstrap_01.png)

#### Bootstrap Sampling Process

```python
lista_de_modelos = []
N_modelos = 10

for i in range(N_modelos):
    X_train_bootstrap, _, y_train_bootstrap, _ = train_test_split(
        X_train, y_train, test_size=0.5, stratify=y_train
    )
    clf = DecisionTreeClassifier(max_depth=None)  # Allow overfitting
    clf.fit(X_train_bootstrap, y_train_bootstrap)
    lista_de_modelos.append(clf)
```

![Bagging Process](./img/bagging_02.png)

#### Ensemble Prediction Aggregation

The manual bagging implementation:

1. **Creates multiple bootstrap samples** from the training data
2. **Trains individual decision trees** on each sample (allowing overfitting)
3. **Averages probabilities** from all models for final prediction
4. **Applies threshold** (0.5) to convert probabilities to binary predictions

#### Performance Comparison

- Individual models show overfitting (high variance)
- Ensemble reduces variance and improves generalization
- Comparison with scikit-learn's `BaggingClassifier`

### 3. Random Forest Implementation

![Random Forest Architecture](./img/random_forest_01.png)

#### Key Features Explored

```python
clf = RandomForestClassifier(
    n_estimators=100,
    max_features='sqrt',
    n_jobs=-1,
    oob_score=True,
    random_state=42
)
```

#### Advanced Random Forest Analysis

**1. Out-of-Bag (OOB) Scoring**

- Built-in cross-validation without separate validation set
- Efficient performance estimation during training

**2. Feature Importance Analysis**

```python
importances = clf.feature_importances_
# Visualization of feature importance rankings
```

**3. Individual Estimator Analysis**

- Examination of individual trees within the forest
- Understanding why Random Forest trees don't achieve 100% training accuracy

**4. Hyperparameter Analysis**

**Validation Curve (Number of Estimators)**

```python
N_estimadores = [1,2,3,4,5,10,25,50,100,250,500,1000]
# Performance tracking across different ensemble sizes
```

The notebook demonstrates:

- How performance improves with more estimators
- Convergence behavior of the ensemble
- Comparison between train, test, and OOB scores

**Learning Curve Analysis**

```python
train_sizes, train_scores, valid_scores = learning_curve(
    clf, X_train, y_train,
    train_sizes=np.linspace(0.0001,1,10),
    scoring='accuracy', cv=5
)
```

Shows how model performance changes with training set size, helping identify:

- Whether more data would improve performance
- Signs of overfitting or underfitting
- Model convergence behavior

### 4. Visualization and Interpretation

#### Decision Boundary Visualization

For 2D feature spaces (MaxTemp vs. Humidity3pm):

```python
# Creates contour plots showing decision boundaries
# Compares individual vs. ensemble decision regions
```

#### Performance Metrics

- **Accuracy scores** for train and test sets
- **Ensemble vs. individual model** comparisons
- **Benchmark comparisons** with naive predictors

## Key Results and Insights

### Performance Improvements

1. **Individual Decision Trees**: High variance, prone to overfitting
2. **Manual Bagging**: Reduced variance through model averaging
3. **Scikit-learn Bagging**: Consistent performance with manual implementation
4. **Random Forest**: Best performance through feature randomization + bagging

### Important Findings

**1. Variance Reduction**

- Ensemble methods effectively reduce the high variance of individual decision trees
- Performance improves significantly compared to single models

**2. Feature Importance**

- Random Forest provides interpretable feature rankings
- Humidity3pm and MaxTemp show highest predictive power for rain prediction

**3. OOB Validation**

- Out-of-bag scores provide reliable performance estimates
- Eliminates need for separate validation sets

**4. Convergence Behavior**

- Performance plateaus around 100-250 estimators
- Diminishing returns beyond optimal ensemble size

## Technical Implementation Details

### Libraries Used

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import accuracy_score
```

### Key Code Patterns

**Bootstrap Sampling**

```python
# Manual bootstrap implementation
X_train_bootstrap, _, y_train_bootstrap, _ = train_test_split(
    X_train, y_train, test_size=0.5, stratify=y_train
)
```

**Probability Averaging**

```python
# Ensemble prediction aggregation
probs_test_pred = np.zeros(y_test.size)
for modelo in lista_de_modelos:
    probs_test_pred += modelo.predict_proba(X_test)[:,1]
probs_test_pred = probs_test_pred / N_modelos
```

**Feature Importance Visualization**

```python
# Random Forest feature importance
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
sns.barplot(x=columns[indices], y=importances[indices])
```

## Learning Objectives Achieved

1. **Manual Ensemble Implementation**: Understanding the mechanics of bagging
2. **Bootstrap Sampling**: Practical implementation of variance reduction
3. **Random Forest Mastery**: Advanced ensemble techniques with feature randomization
4. **Performance Evaluation**: Comprehensive model comparison and validation
5. **Hyperparameter Analysis**: Systematic exploration of model parameters
6. **Visualization**: Decision boundaries and performance curves

## Extensions and Future Work

The notebook concludes with suggestions for further exploration:

1. **Additional Features**: Incorporate more weather variables
2. **Advanced Metrics**: Precision, recall, F1-score, ROC curves
3. **Cross-validation**: More robust performance estimation
4. **Other Ensemble Methods**: Boosting techniques (AdaBoost, Gradient Boosting)
5. **Feature Engineering**: Domain-specific weather features

## Practical Applications

This solution demonstrates real-world machine learning workflow:

- **Data preprocessing** and cleaning strategies
- **Baseline model** establishment for comparison
- **Progressive model complexity** from simple to advanced ensembles
- **Performance visualization** and interpretation
- **Model selection** based on validation techniques

The weather prediction use case showcases how ensemble methods can improve prediction accuracy in practical scenarios where individual models may struggle with high variance.

---

_This notebook provides a comprehensive solution to weather prediction using ensemble methods, demonstrating both theoretical understanding and practical implementation skills in machine learning._
