🏠 Germany Base Rent Prediction Project
Overview

This project predicts the base rent of residential properties in Germany using machine learning. It is built with CatBoost, and the workflow includes preprocessing, feature engineering, hyperparameter tuning, evaluation, and deployment-ready scripts with Streamlit.

The project is structured to allow end-to-end model development: data preprocessing → model training → hyperparameter tuning → evaluation → deployment.

Features

Preprocessing and cleaning of real estate dataset

Feature engineering:

propertyAge (derived from construction year)

livingSpaceRange (categorical bins for living space)

Handling missing values, outliers, and high-cardinality categorical variables

Encoding:

One-hot encoding for low-cardinality features (regio1, condition, interiorQual)

Ordinal encoding for high-cardinality features (regio2, regio3)

Scaling numeric features

Model:

CatBoost Regressor for robust performance with categorical features

Hyperparameter tuning via RandomizedSearchCV

Model evaluation metrics:

MAE, RMSE, R², MAPE

Cross-validation (5-fold)

Feature importance visualization

Deployment-ready Streamlit app

Project Structure
Classical_ML/
│
├── Data/
│   └── immo_data.csv          # Dataset
│
├── app.py                     # Streamlit deployment app
├── evaluate_model.py          # Evaluation, cross-validation, and hyperparameter tuning
├── best_model.pkl             # Baseline trained CatBoost model
├── best_model_tuned.pkl       # Tuned CatBoost model
├── encoder.pkl                # Saved OrdinalEncoder
├── scaler.pkl                 # Saved StandardScaler
├── train_columns.pkl          # Training feature column order
└── README.md                  # Project documentation

Setup Instructions

Clone the repository:

git clone <repo_url>
cd Classical_ML


Create and activate a Python virtual environment:

python -m venv venv
source venv/bin/activate       # Linux/Mac
venv\Scripts\activate          # Windows


Install dependencies:

pip install -r requirements.txt


Ensure the dataset (immo_data.csv) is in the Data folder.

Usage
1. Model Evaluation & Hyperparameter Tuning
python evaluate_model.py


Evaluates baseline model

Performs RandomizedSearchCV hyperparameter tuning

Saves the best model as best_model_tuned.pkl

Displays feature importance and metrics

2. Streamlit Deployment
streamlit run app.py


User-friendly web interface for property rent prediction

Input property features and get predicted base rent

Supports all preprocessing and feature alignment from training

Evaluation Metrics

MAE (Mean Absolute Error)

RMSE (Root Mean Squared Error)

R² (Coefficient of Determination)

MAPE (Mean Absolute Percentage Error)

Cross-validation RMSE

Metrics are printed for both baseline and tuned models.

Dependencies

Python >= 3.8

pandas

numpy

scikit-learn

catboost

matplotlib

seaborn

joblib

pickle

streamlit

Install all dependencies using:

pip install pandas numpy scikit-learn catboost matplotlib seaborn joblib streamlit

Notes

Ensure train_columns.pkl, encoder.pkl, and scaler.pkl are loaded for deployment to match feature preprocessing exactly as in training.

Tuned model best_model_tuned.pkl generally provides better predictions than baseline.

Designed for German residential rental data; may need adjustment for other countries or datasets.

Future Improvements

Hyperparameter tuning with Optuna or Bayesian Optimization

SHAP feature importance for better model explainability

Deployment with Docker or FastAPI for production-grade service

MLflow integration for experiment tracking