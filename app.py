import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from kalman import KalmanRegression

st.set_page_config(page_title="Alpha Research: Pairs Trading", layout="wide")
st.title("⚡ Alpha Research: Statistical Arbitrage Engine")
st.markdown("""
This dashboard simulates a **Pairs Trading Strategy** using a **Kalman Filter**. 
Unlike static linear regression, the Kalman Filter updates the "Hedge Ratio" dynamically 
to adapt to market regime changes.
""")

st.sidebar.header("Strategy Parameters")

ticker1 = st.sidebar.text_input("Stock 1 (The Target)", value="PEP")
ticker2 = st.sidebar.text_input("Stock 2 (The Reference)", value="KO")

start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-01-01"))

z_threshold = st.sidebar.slider("Z-Score Threshold (Entry)", 1.0, 3.0, 2.0, 0.1)
window_size = st.sidebar.slider("Rolling Window (Days)", 10, 60, 30)

@st.cache_data
def get_data(t1, t2, start, end):
    t1 = t1.upper()
    t2 = t2.upper()
    
    try:
        df = yf.download([t1, t2], start=start, end=end, auto_adjust=True)['Close']
        
        if df.empty:
            st.error("No data found. Check your internet or tickers.")
            return None
            
        if t1 not in df.columns or t2 not in df.columns:
            st.error(f"Could not find data for {t1} or {t2}. Check spelling.")
            return None
            
        return df.dropna()

    except Exception as e:
        st.error(f"Error: {e}")
        return None

# --- RUN BUTTON ---
if st.sidebar.button("Run Simulation"):
    
    with st.spinner('Fetching Market Data & Running Kalman Filter...'):
        prices = get_data(ticker1, ticker2, start_date, end_date)
        
        if prices is None or prices.empty:
            st.error("Error: Could not fetch data. Check your tickers.")
        else:
            # --- 1. RUN KALMAN FILTER ---
            kf = KalmanRegression()
            betas = []
            alphas = []
            
            for i in range(len(prices)):
                p1 = prices[ticker1].iloc[i]
                p2 = prices[ticker2].iloc[i]
                state = kf.update(p1, p2)
                alphas.append(state[0])
                betas.append(state[1])
            
            # --- 2. GENERATE SIGNALS ---
            obs_alpha = pd.Series(alphas, index=prices.index)
            obs_beta = pd.Series(betas, index=prices.index)
            
            fair_value = obs_alpha + (obs_beta * prices[ticker2])
            spread = prices[ticker1] - fair_value
            
            # Calculate Z-Score
            spread_mean = spread.rolling(window=window_size).mean()
            spread_std = spread.rolling(window=window_size).std()
            z_score = (spread - spread_mean) / spread_std
            
            # --- 3. BACKTEST (With Transaction Costs) ---
            positions = []
            current_pos = 0
            
            for z in z_score:
                if z > z_threshold:
                    current_pos = -1 # Short
                elif z < -z_threshold:
                    current_pos = 1  # Long
                elif abs(z) < 0.5:
                    current_pos = 0  # Exit
                positions.append(current_pos)
            
            pos_series = pd.Series(positions, index=prices.index)
            
            # PnL Calculation
            spread_change = spread - spread.shift(1)
            daily_pnl = pos_series.shift(1) * spread_change
            
            # Transaction Costs
            cost_per_trade = 0.005 
            trades_made = pos_series.diff().abs().fillna(0)
            daily_pnl_net = daily_pnl - (trades_made * cost_per_trade)
            cumulative_pnl = daily_pnl_net.cumsum()
            
            # Sharpe Ratio
            if daily_pnl_net.std() != 0:
                sharpe = (daily_pnl_net.mean() / daily_pnl_net.std()) * (252**0.5)
            else:
                sharpe = 0.0

            # --- 4. DISPLAY METRICS ---
            col1, col2, col3 = st.columns(3)
            col1.metric("Net Profit", f"${cumulative_pnl.iloc[-1]:.2f}")
            col2.metric("Sharpe Ratio", f"{sharpe:.2f}")
            col3.metric("Total Trades", f"{int(trades_made.sum())}")

            # --- 5. DISPLAY CHARTS ---
            
            # Chart A: Prices
            st.subheader(f"Price Correlation: {ticker1} vs {ticker2}")
            st.line_chart(prices)
            
            # Chart B: Z-Score
            st.subheader("Trading Signals (Z-Score)")
            
            fig_z, ax_z = plt.subplots(figsize=(10, 3))
            ax_z.plot(z_score.index, z_score, label="Z-Score", color='blue', linewidth=1)
            ax_z.axhline(z_threshold, color='red', linestyle='--', label='Short Threshold')
            ax_z.axhline(-z_threshold, color='green', linestyle='--', label='Long Threshold')
            ax_z.axhline(0, color='black', alpha=0.5)
            ax_z.legend(loc='upper left')
            st.pyplot(fig_z)
            
            # Chart C: Performance
            st.subheader("Strategy Performance (Net PnL)")
            st.line_chart(cumulative_pnl)
            
else:
    st.info("👈 Enter tickers in the sidebar and click 'Run Simulation' to start.")