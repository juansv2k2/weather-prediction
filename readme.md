# 🌦️ Weather Prediction Using Ensemble Learning

## Project Overview

A comprehensive machine learning project that predicts whether it will rain tomorrow using advanced ensemble methods. This project demonstrates the progression from simple baseline models to sophisticated ensemble techniques, showcasing how **Bagging** and **Random Forest** algorithms can significantly improve prediction accuracy over individual decision trees.

**Key Achievement**: Developed an ensemble model that outperforms naive baseline predictors by leveraging bootstrap sampling and feature randomization techniques.

## Business Problem

Weather prediction is crucial for agriculture, event planning, and daily decision-making. This project tackles the binary classification challenge of predicting rain occurrence using historical weather patterns from Australian meteorological data.

**Dataset**: [Australian Weather Dataset](https://www.kaggle.com/jsphyg/weather-dataset-rattle-package) (~145k observations)  
**Target**: RainTomorrow (binary: Yes/No)  
**Features**: Temperature, humidity, pressure, wind measurements, and more

### Class Distribution Analysis

The dataset shows a natural imbalance typical in weather prediction:

![Class Distribution](./images/class_distribution.png)

_Approximately 77% "No Rain" vs 23% "Rain" - a realistic representation of Australian weather patterns that creates the foundation for baseline model comparisons._

## 🎯 Project Methodology

### 1. Data Preprocessing & Feature Engineering

**Smart Data Cleaning Strategy:**

- Removed features with >50% missing values (Sunshine, Evaporation, Cloud measurements)
- Eliminated location and date dependencies for model generalization
- Applied correlation analysis to reduce multicollinearity
- Strategic feature selection based on domain knowledge

**Final Feature Set:** 16 numerical weather measurements including MaxTemp, MinTemp, Humidity levels, Pressure readings, and Wind speeds.

### Feature Correlation Analysis

![Correlation Heatmap](./images/correlation_heatmap.png)

_Correlation analysis reveals strong relationships between temperature measurements and pressure readings, guiding strategic feature selection to reduce multicollinearity._

### 2. Baseline Model Establishment

Before implementing complex algorithms, established simple benchmarks:

- **"Always No Rain" Model**: 77.4% accuracy (class frequency baseline)
- **"Always Rain" Model**: 22.6% accuracy

These baselines provide context for evaluating ensemble improvements.

### 3. Manual Bagging Implementation

**Core Innovation**: Built bagging from scratch to demonstrate understanding of ensemble mechanics.

**Process:**

1. **Bootstrap Sampling**: Created 10 diverse training subsets using random sampling with replacement
2. **Weak Learner Training**: Trained individual decision trees (intentionally allowed to overfit)
3. **Prediction Aggregation**: Averaged probabilities across all models
4. **Threshold Application**: Converted averaged probabilities to binary predictions

**Key Insight**: Individual trees showed high variance (~85-95% accuracy range), but ensemble achieved consistent ~82% accuracy.

### 4. Advanced Random Forest Analysis

**Feature Importance Discovery:**

![Feature Importance](./images/feature_importance.png)

_The Random Forest analysis reveals that Humidity3pm and MaxTemp emerge as the strongest predictors, aligning with meteorological domain knowledge and providing actionable insights for weather forecasting._

**Hyperparameter Optimization:**

![Validation Curve](./images/validation_curve.png)

_Performance analysis shows the model plateaus around 100-250 estimators, demonstrating optimal ensemble size for balancing computational efficiency with predictive accuracy._

## 📊 Results & Performance Analysis

### Model Performance Comparison

| Model Type                 | Training Accuracy | Test Accuracy | Key Insight                                 |
| -------------------------- | ----------------- | ------------- | ------------------------------------------- |
| Baseline (No Rain)         | 77.4%             | 77.4%         | Class frequency benchmark                   |
| Individual Decision Tree   | ~95%              | ~82%          | High variance, overfitting                  |
| Manual Bagging (10 trees)  | ~87%              | ~83%          | Variance reduction achieved                 |
| Scikit-learn Random Forest | ~98%              | ~85%          | Best performance with feature randomization |

### Learning Curve Analysis

![Learning Curve](./images/learning_curve.png)

_The learning curve analysis reveals the model benefits from additional data, with training and validation scores converging around 85% accuracy, indicating good generalization without significant overfitting._

### Decision Boundary Visualization

For the two most important features (MaxTemp vs Humidity3pm):

![Decision Boundaries](./images/decision_boundaries.png)

_Random Forest creates sophisticated, non-linear decision boundaries that effectively separate rain and no-rain conditions better than individual decision trees._

### Out-of-Bag (OOB) Validation

**Innovation**: Used OOB scoring for efficient model validation without separate validation sets.

- OOB Score: ~84.8%
- Closely matches cross-validation results
- Demonstrates internal validation reliability

## 🧠 Technical Skills

### Machine Learning

- **Ensemble Methods**: Manual implementation of bagging algorithms
- **Bootstrap Sampling**: Understanding of variance reduction techniques
- **Random Forest Mastery**: Hyperparameter tuning and analysis
- **Model Evaluation**: Validation using multiple metrics
- **Feature Engineering**: Correlation analysis and strategic feature selection

### Data Science

- **EDA Best Practices**: Systematic data exploration and visualization
- **Preprocessing Pipeline**: Handling missing data and feature scaling
- **Baseline Establishment**: Scientific approach to model comparison
- **Performance Visualization**: Learning curves, validation curves, decision boundaries

### Python & Libraries

- **Scikit-learn**: Advanced usage of ensemble algorithms and evaluation metrics
- **Pandas/NumPy**: Efficient data manipulation and numerical computing
- **Matplotlib/Seaborn**: Professional data visualization and interpretation
- **Statistical Analysis**: Correlation analysis and feature importance interpretation
