CODSOFT

Task 1: Customer Churn Prediction

Objective:

The goal of this task was to develop a machine learning model to predict customer churn for a subscription-based service. Churn prediction helps businesses identify customers who are likely to cancel their subscription, enabling them to take proactive measures to retain these customers.

Dataset:

The task utilized a dataset containing historical customer data, including features such as customer demographics, account information, and service usage behavior. The target variable, Exited, indicates whether a customer churned (1) or remained with the service (0).

Steps Involved:

Data Preprocessing:

Handling Missing Values: Missing data was filled using median values for numerical columns and mode for categorical columns.
Feature Selection: Irrelevant columns such as RowNumber, CustomerId, and Surname were dropped.
Encoding Categorical Variables: The categorical columns (Geography, Gender) were encoded using one-hot encoding to convert them into numerical format.
Feature Scaling: All features were scaled using StandardScaler to standardize the data.

Model Building:

Multiple machine learning models were trained and evaluated, including:
Logistic Regression: A simple yet effective model for binary classification.
Random Forest: An ensemble model that builds multiple decision trees and merges them together for better accuracy.
Gradient Boosting: Another ensemble method that builds models sequentially, each new model correcting the errors made by the previous ones.

Model Evaluation:

The models were evaluated using various metrics such as accuracy, precision, recall, F1-score, and confusion matrix. This helped in identifying the best-performing model.

Model Selection and Saving:

The Gradient Boosting model was selected as the best-performing model based on its accuracy and other evaluation metrics.
The final model was saved using joblib for future predictions.

Results:

The Gradient Boosting model achieved the highest accuracy and was able to effectively identify customers who are likely to churn. This model can be used by the business to predict churn and take preventive actions to retain customers


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
