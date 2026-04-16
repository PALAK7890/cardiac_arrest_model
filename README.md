# Cardiac Arrest Risk Prediction System

## Project Overview
This project is an end-to-end machine learning system designed to predict the risk of cardiac arrest using patient health indicators such as age, BMI, blood pressure, cholesterol, and lifestyle-related factors. A TensorFlow-based neural network model is trained on structured healthcare data and deployed using Streamlit for real-time predictions.

## Objective
To build a predictive system that classifies individuals into low, medium, and high cardiac risk categories to support early health risk identification.

## Features
- End-to-end machine learning pipeline from data preprocessing to deployment  
- Neural network model built using TensorFlow/Keras  
- Class imbalance handling using class weights  
- Real-time prediction through Streamlit web application  
- Risk stratification into low, medium, and high categories  

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, TensorFlow, Keras, Streamlit, Joblib

## Data Preprocessing
- Handling missing values  
- Feature scaling using StandardScaler  
- Encoding categorical variables  
- Train-test split (80/20)  

## Model Architecture
- Input layer with health-related features  
- Dense layer (128 units, ReLU)  
- Batch Normalization  
- Dropout (0.3)  
- Dense layer (64 units, ReLU)  
- Dropout (0.2)  
- Dense layer (32 units, ReLU)  
- Output layer (Sigmoid activation)  

## Evaluation Metrics
Accuracy, Precision, Recall, F1-score, ROC-AUC, Confusion Matrix, Precision-Recall Curve

## Project Structure
Cardiac_Model/
- app.py
- heart_model.h5
- scaler.pkl
- README.md

## Future Improvements
- Integration of model explainability using SHAP  
- Comparison with classical machine learning models  
- Enhanced dashboard-based UI  
- Cloud deployment for public access
