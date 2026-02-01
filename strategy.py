import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from kalman import KalmanRegression

def run_strategy():
    tickers = ['PEP', 'KO']
    start_date = '2020-01-01'
    end_date = '2024-01-01'

    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)
    prices = data['Close'].dropna()

    kf = KalmanRegression()
    betas = []
    alphas = []

    for i in range(len(prices)):
        price_pep = prices['PEP'].iloc[i]
        price_ko = prices['KO'].iloc[i]
        state = kf.update(price_pep, price_ko)
        alphas.append(state[0])
        betas.append(state[1])

    obs_alpha = pd.Series(alphas, index=prices.index)
    obs_beta = pd.Series(betas, index=prices.index)

    fair_value = obs_alpha + (obs_beta * prices['KO'])
    spread = prices['PEP'] - fair_value

    spread_mean = spread.rolling(window=30).mean()
    spread_std = spread.rolling(window=30).std()
    z_score = (spread - spread_mean) / spread_std

    positions = []
    current_pos = 0

    for z in z_score:
        if z > 2.0:
            current_pos = -1
        elif z < -2.0:
            current_pos = 1
        elif abs(z) < 0.5:
            current_pos = 0
        positions.append(current_pos)

    pos_series = pd.Series(positions, index=prices.index)

    spread_change = spread - spread.shift(1)
    daily_pnl = pos_series.shift(1) * spread_change

    cost_per_trade = 0.005 
    
    trades_made = pos_series.diff().abs().fillna(0)
    
    daily_pnl_after_cost = daily_pnl - (trades_made * cost_per_trade)

    cumulative_pnl = daily_pnl_after_cost.cumsum()

    sharpe_ratio = (daily_pnl_after_cost.mean() / daily_pnl_after_cost.std()) * (252**0.5)

    print(f"Final PnL: ${cumulative_pnl.iloc[-1]:.2f}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

    plt.figure(figsize=(12, 10))

    plt.subplot(3, 1, 1)
    plt.plot(prices['PEP'], label='PEP')
    plt.plot(prices['KO'], label='KO')
    plt.title('Prices')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(z_score, label='Z-Score')
    plt.axhline(2.0, color='red', linestyle='--')
    plt.axhline(-2.0, color='green', linestyle='--')
    plt.axhline(0, color='black')
    plt.title('Trading Signals')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(daily_pnl.cumsum(), color='green', alpha=0.3, label='Raw Profit (Ideal)')
    plt.plot(cumulative_pnl, color='blue', label='Net Profit (After Costs)')
    plt.title(f'Cumulative PnL (Sharpe Ratio: {sharpe_ratio:.2f})')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_strategy()