Medical Diagnosis using Supervised Machine Learning

1.Project Overview

This project implements a supervised machine learning classification model to predict whether a patient has diabetes based on medical diagnostic measurements. The goal is to apply core concepts of Supervised Learning such as data preprocessing, model training, evaluation, and interpretation.

This project is inspired by the Supervised Machine Learning module from Andrew Ng’s Machine Learning course on Coursera.

2.Problem Statement

Early detection of diabetes is crucial for preventing severe health complications.
Using patient health parameters, this project predicts the likelihood of diabetes using machine learning classification techniques.

3.Dataset Information

Dataset: Pima Indians Diabetes Dataset
Source: Public ML dataset (Kaggle / UCI Repository)
Number of features: 8
Target variable: Outcome
0 → Non-diabetic
1 → Diabetic
Features Used:
Pregnancies
Glucose
BloodPressure
SkinThickness
Insulin
BMI
DiabetesPedigreeFunction
Age

4.Technologies & Tools Used

Python
NumPy
Pandas
Matplotlib
Seaborn
Scikit-learn
VS Code

5.Machine Learning Approach

Type: Supervised Learning
Algorithm Used: Logistic Regression
Steps Followed:
Data loading and inspection
Handling missing values
Feature selection
Train-test split
Model training
Model evaluation using accuracy and classification report

6.Model Performance

Accuracy: ~75%
Evaluation Metrics:
Precision
Recall
F1-score
Confusion Matrix

The results show that the model can reasonably classify diabetic and non-diabetic patients using supervised learning techniques.