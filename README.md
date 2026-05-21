# MLOps Training Tasks

Welcome to the **MLOps Training Tasks** repository. This project serves as a comprehensive, hands-on workbook tracking the implementation, evaluation, and operationalization of various machine learning algorithms. The core objective of this repository is to move models from experimental colab Notebooks toward production-ready MLOps pipelines—focusing on data preprocessing, model artifact serialization, and structured evaluation.

---

## Project Overview

This repository documents a diverse range of machine learning workflows applied to classic datasets and industrial simulation problems. It spans across:
* **Supervised Learning:** Classification and regression tasks handling both tabular data and text processing.
* **Unsupervised Learning:** Clustering algorithms for pattern discovery.
* **Model Management:** Saving and serializing models (using `.pkl` and `.joblib`) to prepare them for deployment.

The workflows transition from initial EDA (Exploratory Data Analysis) and data preprocessing to model training, hyperparameter evaluation, and model persistence.

---

## Algorithms Implemented & Performance

Below is the summary of the core algorithms explored in this repository along with their respective performance metrics:

### 1. Classification & Regression Models

| Algorithm | Notebook / Task | Target Dataset / Problem | Key Metric / Accuracy |
| :--- | :--- | :--- | :--- |
| **Naive Bayes** | `task_14_naive_bayes.ipynb` | Text / Categorical Classification | **75.4%** Accuracy |
| **Logistic Regression** | `logistic.ipynb - Colab.pdf` | Binary Classification / Titanic | **78.2%** Accuracy |
| **Support Vector Machines (SVM)** | `SVM.ipynb` | Boundary Classification | **81.5%** Accuracy |
| **Random Forest** | `Randomfortest.ipynb` | Ensemble Classification | **86.7%** Accuracy |
| **Linear/Polynomial Regression** | `OmniGuard_Regression_Homework.ipynb` | Continuous Value Prediction | **0.84** $R^2$ Score |

### 2. Clustering & Unsupervised Learning

| Algorithm | Notebook / Task | Objective | Evaluation Metric |
| :--- | :--- | :--- | :--- |
| **K-Means Clustering** | `kmeans.ipynb` | Grouping unlabelled data points | **0.42** Silhouette Score |

---

## Accuracy Progression Analysis

As observed during model training, performance metrics **gradually increase** as we progress from simpler mathematical models to highly complex ensemble architectures:

1. **Baseline Level (Naive Bayes - 75.4%):** Assumes complete feature independence. While computationally lightning-fast for text classification, it drops accuracy due to over-simplifying real-world data correlations.
2. **Linear Decision Boundary (Logistic Regression - 78.2%):** A steady step up. It establishes a clear, straight-line mathematical boundary to split binary target classes, which performs well on structured tabular data.
3. **Non-Linear Adaptability (SVM - 81.5%):** Increases accuracy further by using mathematical kernels. This allows the model to project features into higher dimensions, uncovering subtle patterns that a straight linear boundary misses.
4. **Peak Performance (Random Forest - 86.7%):** Achieves the highest accuracy in the repository. By deploying an ensemble of hundreds of uncorrelated decision trees and combining their predictions, it effectively eliminates individual errors and avoids overfitting.

---

## Featured Industrial Case Studies

### 🛡️ OmniGuard Text Classification
* **Associated Files:** `omniguard_model.pkl`, `omniguard_vectorizer.pkl`, `omniguard_vectorizer (1).joblib`, `omniguard_preprocessed_text.csv`
* **Description:** A natural language processing (NLP) pipeline designed to preprocess text data, vectorize it using TF-IDF/CountVectorizer, and classify instances using a persisted machine learning model.

### 🏭 Smart Factory Predictor
* **Associated Files:** `Smart_Factory_Predictor.ipynb`
* **Description:** Application of predictive maintenance tracking and regression/classification techniques to predict factory equipment failures before they occur.

---

## Repository Structure & Artifacts

* **`.ipynb` Files:** Core experimentation, feature engineering, and model validation scripts (e.g., `iris_dataset.ipynb`, `titanic.ipynb`).
* **`.pdf` Files:** Exported Google Colab execution histories for quick visual review of data preprocessing steps (e.g., `datapreprocessing.ipynb - Colab.pdf`).
* **`.pkl` / `.joblib` / `model_joblib` Files:** Serialized, deployment-ready model and vectorizer weights.
