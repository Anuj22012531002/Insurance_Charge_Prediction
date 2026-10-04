# 💰 Insurance Charge Prediction using Machine Learning

## 📌 Project Overview

Insurance Charge Prediction is an end-to-end Machine Learning regression project developed to predict the estimated insurance charges of customers based on their demographic, medical, policy, and lifestyle-related information.

The project follows a complete Machine Learning workflow including:

- Data Collection
- Data Cleaning
- Data Preprocessing
- Exploratory Data Analysis
- Feature Engineering
- Categorical Encoding
- Train-Test Split
- Model Training
- Model Evaluation
- Hyperparameter Tuning
- Model Comparison
- Feature Importance Analysis
- Model Serialization
- Streamlit Deployment

The final selected model is **XGB2_Reg (Tuned XGBoost Regressor)**.

---

# 🎯 Project Objective

The main objective of this project is to build a Machine Learning regression model capable of accurately predicting insurance charges using customer medical, demographic, policy, and lifestyle information.

### Key Objectives

- Analyze the factors affecting insurance charges.
- Perform data preprocessing and feature engineering.
- Train multiple Machine Learning regression algorithms.
- Compare model performance using multiple evaluation metrics.
- Tune the best-performing models using RandomizedSearchCV.
- Select the final model based on test performance.
- Save the trained model using Pickle.
- Deploy the model using Streamlit.

---

# 📊 Dataset Features

The dataset contains customer, healthcare, insurance policy, and lifestyle-related information.

### Features Used

| Feature | Description |
|---|---|
| `insurance_plan` | Type of insurance plan |
| `coverage_amount` | Insurance coverage amount |
| `hospital_visits` | Number of hospital visits |
| `doctor_visits` | Number of doctor visits |
| `prescription_count` | Number of prescriptions |
| `emergency_visits` | Number of emergency visits |
| `claim_count` | Number of insurance claims |
| `annual_medical_expenses` | Annual medical expenses |
| `policy_tenure_years` | Policy tenure in years |
| `preventive_checkup` | Preventive checkup status |
| `age` | Customer age |
| `sex` | Customer gender |
| `bmi` | Body Mass Index |
| `children` | Number of children |
| `smoker` | Smoking status |
| `region` | Customer region |
| `income_level` | Income category |
| `exercise_level` | Exercise level |
| `family_history` | Family medical history |
| `occupation` | Customer occupation |

### Target Variable

```text
charges
