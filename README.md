# Telecom Churn Prediction Model

## Overview
This repository contains a machine learning pipeline for predicting customer churn in a telecom dataset using a RandomForestClassifier. The model is optimized for high recall (83.16% on evaluation data) to identify at-risk customers, demonstrating skills in feature engineering, pipeline construction, hyperparameter tuning, and handling imbalanced data.

## Features
- **Dataset**: Telecom customer data with ~14% churn rate, including call minutes, customer service calls, and plan details.
- **Pipeline**:
  - **Feature Engineering**: Creates `Total minutes`, `Total calls`, and `High customer service calls` (>4 calls).
  - **Preprocessing**: One-hot encoding for `State` and `Area code`, standard scaling for numeric features.
  - **Feature Selection**: Uses `SelectFromModel` with RandomForestClassifier.
  - **Model**: RandomForestClassifier with balanced class weights and a custom threshold (0.4).
- **Evaluation**: Metrics include recall, precision, and confusion matrix, with a precision-recall curve.

## Results
- **Test Set** (534 samples, ~15% churn):
  - Recall for churn: 81.01%
  - Precision for churn: 50.00%
  - Confusion Matrix: `[[392, 63], [15, 64]]`
- **Evaluation Set** (667 samples, ~14.24% churn):
  - Recall for churn: 83.16%
  - Precision for churn: 46.47%
  - Confusion Matrix: `[[481, 91], [16, 79]]`

## Installation
