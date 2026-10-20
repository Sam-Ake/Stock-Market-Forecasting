import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# Page Configuration
st.set_page_config(
    page_title="Stock Market Time-Series Forecaster",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Stock Market Time-Series Forecasting App")
st.markdown(
    "Explore live stock prices, analyze technical moving averages, and predict"
    " future closing prices using a trained Random Forest model."
)


# Load Model
@st.cache_resource
def load_model():
  try:
    return joblib.load("stock_rf_model.pkl")
  except Exception as e:
    return None


model = load_model()

if model is None:
  st.warning(
      "⚠️ Model file 'stock_rf_model.pkl' not found. Please ensure it's in the"
      " root directory."
  )

# Sidebar Configuration for Live Ticker Input
st.sidebar.header("⚙️ Configuration")
ticker_symbol = st.sidebar.text_input(
    "Enter Stock Ticker (e.g., AAPL, TSLA, MSFT)", value="AAPL"
).upper()


# Load Live Data Function via yfinance (2023 to Present)
@st.cache_data
def load_live_data(ticker):
  try:
    df = yf.download(ticker, start="2023-01-01", end="2026-10-20")
    if isinstance(df.columns, pd.MultiIndex):
      df.columns = df.columns.get_level_values(0)
    df = df[["Close"]].dropna()
    return df
  except Exception as e:
    return None


df = load_live_data(ticker_symbol)

if df is not None and not df.empty:
  # Main Layout Tabs
  tab1, tab2 = st.tabs(["📊 Exploratory Data & Trends", "🔮 Live Forecasting"])

  with tab1:
    st.subheader(f"Historical Price & Moving Averages: {ticker_symbol}")

    # Plotly Interactive Chart
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Close"],
            mode="lines",
            name="Close Price",
            line=dict(color="royalblue", width=1.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Close"].rolling(window=30).mean(),
            mode="lines",
            name="30-Day MA",
            line=dict(color="darkorange", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Close"].rolling(window=200).mean(),
            mode="lines",
            name="200-Day MA",
            line=dict(color="forestgreen", width=2),
        )
    )

    fig.update_layout(
        title=f"{ticker_symbol} Daily Closing Prices (2023 - Present)",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_white",
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Dataset Quick Stats
    col1, col2, col3 = st.columns(3)
    col1.metric("Latest Close Price", f"${float(df['Close'].iloc[-1]):.2f}")
    col2.metric("Total Records", len(df))
    col3.metric(
        "Date Range", f"{df.index.min().date()} to {df.index.max().date()}"
    )

  with tab2:
    st.subheader("🔮 Predict Future Stock Close Price")
    st.markdown(
        "Input recent lag prices and rolling means to generate a prediction via"
        " the model."
    )

    last_close = float(df["Close"].iloc[-1])

    col1, col2 = st.columns(2)
    with col1:
      lag_1 = st.number_input("Lag 1 (Yesterday Close)", value=last_close)
      lag_2 = st.number_input(
          "Lag 2 (2 Days Ago)", value=float(df["Close"].iloc[-2])
      )
      lag_3 = st.number_input(
          "Lag 3 (3 Days Ago)", value=float(df["Close"].iloc[-3])
      )
      lag_4 = st.number_input(
          "Lag 4 (4 Days Ago)", value=float(df["Close"].iloc[-4])
      )

    with col2:
      lag_5 = st.number_input(
          "Lag 5 (5 Days Ago)", value=float(df["Close"].iloc[-5])
      )
      rolling_mean_7 = st.number_input(
          "7-Day Rolling Mean",
          value=float(df["Close"].shift(1).rolling(7).mean().iloc[-1]),
      )
      rolling_mean_30 = st.number_input(
          "30-Day Rolling Mean",
          value=float(df["Close"].shift(1).rolling(30).mean().iloc[-1]),
      )

    if st.button("Predict Close Price", type="primary"):
      if model is not None:
        features = np.array([[
            lag_1,
            lag_2,
            lag_3,
            lag_4,
            lag_5,
            rolling_mean_7,
            rolling_mean_30,
        ]])
        prediction = model.predict(features)[0]
        st.success(f"### Predicted Next Close Price: ${prediction:.2f}")
      else:
        st.error("Model is not loaded.")
else:
  st.error(
      "Could not fetch data for this ticker symbol. Please check the ticker"
      " and try again."
  )