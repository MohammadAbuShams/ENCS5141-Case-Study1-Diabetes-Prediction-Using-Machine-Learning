# Diabetes Prediction Using Machine Learning

## Overview
This project demonstrates the impact of **data cleaning** and **feature engineering** on the performance of machine learning models for diabetes prediction. The study also includes a **comparative analysis** of three popular classification models: **Random Forest (RF)**, **Support Vector Machine (SVM)**, and **Multilayer Perceptron (MLP)**. The goal is to identify the best model for predicting diabetes while achieving a balance between accuracy and computational efficiency.

---

## Objectives
### Part 1: Data Cleaning and Feature Engineering
- Explore and preprocess the dataset to address issues such as missing values and outliers.
- Encode categorical features and scale numerical features for consistency.
- Use **Principal Component Analysis (PCA)** for dimensionality reduction.
- Compare the performance of models trained on **raw data** and **preprocessed data**.

### Part 2: Comparative Analysis of Classification Techniques
- Train and evaluate RF, SVM, and MLP models on the preprocessed dataset.
- Analyze each model in terms of:
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1 Score**
  - **Training Time**
- Determine the model that provides the **best balance** between accuracy and computational time.

---

## Dataset
The **Diabetes Dataset** includes various health indicators, such as:
- Age
- BMI
- Blood Pressure
- Glucose Levels

### Preprocessing Steps:
1. **Handle Missing Values**: Replaced missing values with column means.
2. **Outlier Treatment**: Capped outliers within the 1st and 99th percentiles.
3. **Feature Scaling**: Standardized numerical features for consistency.
4. **One-Hot Encoding**: Converted categorical variables into binary columns.
5. **Dimensionality Reduction**: Applied PCA to retain 95% variance.

---

## Methodology

### 1. Data Cleaning & Preprocessing
The dataset was preprocessed to improve model performance:
- Missing values and outliers were handled.
- Features were standardized and encoded.
- PCA was applied to reduce dimensions while preserving key information.

### 2. Model Training & Evaluation
Three models were trained and evaluated on the preprocessed dataset:
- **Random Forest (RF)**
- **Support Vector Machine (SVM)**
- **Multilayer Perceptron (MLP)**

### 3. Comparative Analysis
The models were evaluated on:
- **Prediction Metrics**: Accuracy, Precision, Recall, and F1 Score.
- **Efficiency**: Training time.
A weighted scoring method (70% accuracy, 30% training time) was used to determine the best model.

---

## Key Results
### Preprocessed vs. Raw Data Comparison
| Metric              | Preprocessed Data | Raw Data   |
|---------------------|------------------|-----------|
| **Accuracy**        | 83.23%           | 90.05%    |
| **Precision**       | 83.41%           | 90.45%    |
| **Recall**          | 83.26%           | 90.08%    |
| **Training Time**   | 65.06 seconds    | 12.82 seconds |

The raw data performed better in this case due to the model's ability to handle unprocessed features efficiently.

### Comparative Analysis of Models
| Metric              | Random Forest    | SVM        | MLP        |
|---------------------|------------------|------------|------------|
| **Accuracy**        | 83.23%           | 85.31%     | 88.46%     |
| **Precision**       | 83.41%           | 85.45%     | 88.55%     |
| **Recall**          | 83.26%           | 85.32%     | 88.48%     |
| **F1 Score**        | 83.26%           | 85.29%     | 88.43%     |
| **Training Time**   | 64.46 seconds    | 35.13 seconds | 163.36 seconds |
| **Balance Score**   | 82.21%           | 97.51%     | 76.45%     |

### Conclusion
- **SVM** was identified as the best model, with a balance score of **97.51%**, offering strong accuracy and shorter training time compared to MLP.
- **MLP** achieved the highest accuracy (88.46%) but required the longest training time (163.36 seconds).
- **Random Forest** trained faster than MLP but had the lowest accuracy (83.23%).

---
