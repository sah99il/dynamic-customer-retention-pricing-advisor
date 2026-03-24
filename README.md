Customer Retention & Pricing Advisor

This project focuses on predicting customer churn and providing simple, actionable retention strategies using a machine learning pipeline with a backend API and a user interface.

---

Overview

The goal of this project is to identify customers who are likely to churn and suggest appropriate actions to retain them.

The system is built as an end-to-end pipeline that includes data preprocessing, model training, backend integration, and a frontend interface for user interaction.

---

Problem Statement

Customer churn is a critical challenge in subscription-based businesses. This project aims to:

- Predict the likelihood of a customer leaving
- Provide basic recommendations to reduce churn

---

Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- FastAPI (backend)
- Streamlit (frontend)

---

Project Structure

project/
│
├── app.py                # Streamlit UI
├── api.py                # FastAPI backend
├── requirements.txt
│
├── models/
│   ├── model.pkl
│   └── features.pkl
│
└── src/
    ├── preprocess.py
    └── train.py

---

Workflow

1. Data preprocessing and feature engineering
2. Model training using stratified K-fold cross-validation
3. Model selection based on ROC-AUC score
4. Serving predictions using FastAPI
5. Building a simple user interface using Streamlit

---

Model Details

- Models used:
  
  - Logistic Regression
  - Random Forest

- Final model selected based on performance

- Evaluation metric: ROC-AUC (~0.84)

---

Business Logic

Based on predicted churn probability:

- Greater than 0.7 → High risk (suggest retention strategies such as discounts)
- Between 0.4 and 0.7 → Medium risk (engagement recommended)
- Less than 0.4 → Low risk

---

How to Run Locally

1. Install dependencies

pip install -r requirements.txt

2. Start FastAPI backend

uvicorn api:app --reload

3. Run Streamlit UI

streamlit run app.py

---

Demo

The application allows users to input customer details and returns:

- Churn probability
- Risk category
- Suggested action

---

Key Highlights

- End-to-end machine learning pipeline
- Separation of backend and frontend
- Practical focus on business use-case
- Emphasis on simplicity and clarity

---

Learnings

- Feature engineering and data preprocessing
- Model evaluation using cross-validation
- Building API-based ML systems
- Handling integration between UI and backend

---

Future Improvements

- More advanced feature engineering
- More models can be tested to imporve accuracy 
- Hyperparameter tuning
- Deployment using containerization
- Improved recommendation logic

---

Author

This project was developed as part of hands-on learning in machine learning and system design.