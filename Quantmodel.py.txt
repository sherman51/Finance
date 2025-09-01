import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =====================
# Strategy Definitions
# =====================

class BaseStrategy:
    def run(self, data):
        raise NotImplementedError

class MomentumStrategy(BaseStrategy):
    def run(self, data):
        data["Signal"] = np.where(data["Close"] > data["Close"].rolling(20).mean(), 1, 0)
        return data

class MeanReversionStrategy(BaseStrategy):
    def run(self, data):
        sma = data["Close"].rolling(20).mean()
        std = data["Close"].rolling(20).std()
        lower_band = sma - 2 * std
        data["Signal"] = np.where(data["Close"] < lower_band, 1, 0)
        return data

# =====================
# Regime Detection
# =====================

def detect_market_regime(data):
    vol = data["Close"].pct_change().rolling(20).std().iloc[-1]
    trend = data["Close"].iloc[-1] > data["Close"].rolling(50).mean().iloc[-1]
    
    if vol > 0.02 and not trend:
        return "High Volatility / Sideways"
    elif trend:
        return "Trending"
    else:
        return "Low Volatility"

# =====================
# Strategy Selector
# =====================

def select_strategy(regime):
    if regime == "Trending":
        return MomentumStrategy(), "MomentumStrategy"
    elif regime == "High Volatility / Sideways":
        return MeanReversionStrategy(), "MeanReversionStrategy"
    else:
        return MomentumStrategy(), "MomentumStrategy (Default)"

# =====================
# Backtest Logic
# =====================

def backtest(data):
    data["Return"] = data["Close"].pct_change()
    data["Strategy Return"] = data["Signal"].shift(1) * data["Return"]
    data["Equity Curve"] = (1 + data["Strategy Return"]).cumprod()
    data.dropna(inplace=True)
    return data

# =====================
# Streamlit UI
# =====================

st.set_page_config(page_title="Adaptive Quant Model", layout="wide")

st.title("📊 Adaptive Quant Strategy Dashboard")

# User input
ticker = st.text_input("Enter Ticker", "AAPL")
start = st.date_input("Start Date", pd.to_datetime("2022-01-01"))
end = st.date_input("End Date", pd.to_datetime("today"))

# Fetch data
@st.cache_data
def get_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    return df

if st.button("Run Strategy"):
    data = get_data(ticker, start, end)
    st.subheader("Market Regime Detection")
    regime = detect_market_regime(data)
    st.success(f"Detected Regime: **{regime}**")

    strategy, strategy_name = select_strategy(regime)
    st.info(f"Selected Strategy: **{strategy_name}**")

    data = strategy.run(data)
    data = backtest(data)

    st.subheader("Equity Curve")
    fig, ax = plt.subplots()
    ax.plot(data.index, data["Equity Curve"], label="Strategy")
    ax.set_title("Strategy Equity Curve")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Performance Metrics")
    total_return = (data["Equity Curve"].iloc[-1] - 1) * 100
    sharpe = data["Strategy Return"].mean() / data["Strategy Return"].std() * np.sqrt(252)

    st.metric("Total Return (%)", f"{total_return:.2f}")
    st.metric("Sharpe Ratio", f"{sharpe:.2f}")

    st.subheader("Raw Data")
    st.dataframe(data.tail(20))
