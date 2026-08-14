# Explainable AI Credit Default Prediction

An Explainable AI web application that uses an XGBoost machine learning model to predict credit card default risk and provides user-friendly explanations for the prediction using SHAP.

## 🌐 Live Demo

https://xai-loan-default.onrender.com

## ✨ Features

- Credit default risk prediction
- Default probability estimation
- XGBoost machine learning model
- SHAP-based Explainable AI
- User-friendly explanations
- Interactive web interface
- Deployed using Render

## 🛠️ Tech Stack

- Python
- Flask
- XGBoost
- SHAP
- Pandas
- NumPy
- Scikit-learn
- Joblib
- HTML
- CSS
- JavaScript
- Render

## 📁 Project Structure

```text
XAI_Loan_Default/
│
├── app.py
├── index.html
├── xgboost_credit_default_model.pkl
├── requirements.txt
├── .python-version
├── .gitignore
└── README.md
````

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Rajinikanth-Kakarla/XAI_Loan_Default.git
cd XAI_Loan_Default
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the application

Windows:

```bash
py app.py
```

Linux/macOS:

```bash
python app.py
```

### 4. Open in browser

```text
http://127.0.0.1:5000
```

## 🤖 Machine Learning

The application uses a trained XGBoost classifier to estimate the probability of credit card default.

The model uses customer information including:

* Credit limit
* Age
* Education
* Marriage status
* Repayment history
* Bill amounts
* Payment amounts

## 🔍 Explainable AI

SHAP is used to identify the features that have the greatest influence on each prediction.

The technical SHAP results are converted into simple, user-friendly explanations such as:

* Recent repayment behaviour increased risk
* Recent payment activity is helping
* Previous bill amount is influencing the prediction

## ☁️ Deployment

The application is deployed on Render.

### Build Command

```bash
pip install -r requirements.txt
```

### Start Command

```bash
gunicorn app:app
```

### Live Application

[https://xai-loan-default.onrender.com](https://xai-loan-default.onrender.com)

## ⚠️ Disclaimer

This project is for educational and demonstration purposes.

The predictions should not be used as the sole basis for real financial or credit decisions.

## 👨‍💻 Author

**Rajinikanth Kakarla**

```

**This is what I'd actually put on your GitHub.** It gives someone enough information to understand and run the project without turning the README into your project report.
```
