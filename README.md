CODSOFT

TASK1: Movie Genre Classification

Overview

This project focuses on building a machine learning model that predicts the genre of a movie based on its plot summary or other textual information. The model uses natural language processing (NLP) techniques to analyze text and classify it into genres such as Action, Drama, Comedy, etc.

Features

Data Preprocessing: Clean and preprocess the movie plot summaries.

Feature Extraction: Use TF-IDF and word embeddings for text vectorization.

Model Training: Implement and train multiple models, including Naive Bayes, Logistic Regression, and SVM.

TASK2: Credit Card Fraud Detection Using Machine Learning:

Overview:
This repository contains the code and resources for a machine learning project aimed at detecting fraudulent credit card transactions. The project uses various machine learning algorithms to classify transactions as either legitimate or fraudulent, helping to prevent financial losses due to fraud.

Dataset:
The project utilizes the fraudTrain.csv and fraudTest.csv datasets, which contain anonymized credit card transaction data, including features such as transaction amount, location, and merchant details. The data is split into training and testing sets to evaluate the performance of the models.

Project Structure:
fraudTrain.csv: Training dataset containing transaction details.
fraudTest.csv: Testing dataset for evaluating model performance.
credit_card_fraud_detection.ipynb: Jupyter notebook with the complete code for data preprocessing, model training, evaluation, and saving the best model.
credit_card_fraud_model.pkl: Serialized model file for making predictions on new data.
Features and Techniques

Data Preprocessing:

Handling missing values.
Feature engineering from date-time data.
Encoding categorical variables using One-Hot Encoding.
Scaling numerical features using StandardScaler.

Machine Learning Models:

Logistic Regression
Decision Tree
Random Forest

SMOTE: Applied for handling class imbalance to improve model performance.

Evaluation Metrics:

Accuracy
Precision
Recall
F1-Score
Confusion Matrix
