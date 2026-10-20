# 📈 Stock Market Time-Series Forecasting Application

An end-to-end stock market forecasting and technical analysis web application built with **Python**, **FastAPI**, **Streamlit**, and **Random Forest Regression**. It fetches live historical market data via `yfinance`, performs time-series feature engineering (lags and rolling means), and provides both a REST API and an interactive web dashboard.

---

## 🚀 Features
- **Live Data Integration:** Fetches daily closing prices dynamically for any stock ticker using `yfinance`.
- **Technical Analysis:** Interactive Plotly charts displaying Close Price alongside **30-Day** and **200-Day Moving Averages**.
- **Machine Learning Inference:** Predicts future closing prices using a trained **Random Forest Regressor** evaluated via walk-forward time-series cross-validation (~2.62% MAPE).
- **Dual Interface:** 
  - **FastAPI Backend:** Serving model predictions via JSON endpoints (`/forecast`).
  - **Streamlit Frontend:** Interactive user dashboard for data exploration and live inference.

---

## 🛠️ Tech Stack
- **Language:** Python 3.10+
- **Data & ML:** Pandas, NumPy, Scikit-Learn, Statsmodels, YFinance, Joblib
- **Backend:** FastAPI, Uvicorn
- **Frontend & Viz:** Streamlit, Plotly

---

## 📦 Project Structure
```text
stock-market-dataset/
│
├── .venv/                  # Python virtual environment
├── app.py                  # FastAPI backend server
├── streamlit_app.py        # Streamlit interactive frontend dashboard
├── stock_rf_model.pkl      # Trained Random Forest regression model
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation