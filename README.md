# Statistical Arbitrage with Kalman Filters

A quantitative trading strategy that identifies mean-reversion opportunities between cointegrated assets (Pairs Trading). Unlike traditional approaches that use static linear regression, this project utilizes a **Kalman Filter** to dynamically estimate the hedge ratio ($\beta$) in real-time, allowing the model to adapt to structural market changes.

## 🚀 Overview

| | |
|---|---|
| **Strategy** | Statistical Arbitrage / Pairs Trading |
| **Asset Pair** | Pepsi (PEP) and Coca-Cola (KO) |
| **Algorithm** | Linear Kalman Filter for dynamic state estimation |
| **Signal Generation** | Rolling Z-Score of the spread |
| **Performance** | Backtested on 4 years of real market data (2020-2024) |

## 🧠 The Logic

### 1. The Dynamic Hedge Ratio

Standard pairs trading assumes the relationship between two stocks is constant:

$$Price_Y = \alpha + \beta \times Price_X + \epsilon$$

In reality, this relationship drifts over time due to fundamental changes in the companies. This project uses a **Kalman Filter** to treat $\alpha$ and $\beta$ as "hidden states" that are updated with every new price observation.

### 2. Signal Generation (Z-Score)

We calculate the **Spread** between the actual price and the predicted "Fair Value":

$$Spread = Price_{Actual} - (\alpha_{Kalman} + \beta_{Kalman} \times Price_{Reference})$$

We then normalize this spread into a **Z-Score** using a 30-day rolling window to identify statistical anomalies:

| Z-Score | Action |
|---------|--------|
| Z > 2.0 | Asset is overvalued → **Short Entry** |
| Z < -2.0 | Asset is undervalued → **Long Entry** |
| Z ~ 0 | Spread has reverted to mean → **Exit** |

## 🛠️ Tech Stack

- **Python** - Core logic and simulation
- **NumPy** - Matrix multiplication and linear algebra for the Kalman Filter
- **Pandas** - Time-series data manipulation and rolling window calculations
- **Matplotlib** - Visualization of prices, hedge ratios, and equity curves
- **yfinance** - Real-time market data API

## 📂 Project Structure

```
Alpha-Research-Kalman/
├── kalman.py         # Custom KalmanRegression class implementation
├── strategy.py       # Main script: Data fetching, signal generation, and backtesting
├── requirements.txt  # Project dependencies
└── README.md         # Project documentation
```

## ⚡ Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Alpha-Research-Kalman.git
   cd Alpha-Research-Kalman
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the strategy**
   ```bash
   python strategy.py
   ```

## 📊 Results

The script generates a comprehensive dashboard showing:

- **Price History** - Visual confirmation of correlation
- **Trading Signals** - The Z-Score oscillator with ±2.0 thresholds
- **Cumulative PnL** - The equity curve showing the strategy's historical performance

---

> ⚠️ **Disclaimer:** This project is for educational and research purposes only. It does not constitute financial advice. Algorithmic trading involves significant risk.