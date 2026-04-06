# Energy Price Forecasting Model

**Author:** David Darkwa  
**Major:** B.S. Energy Business Finance, Penn State (May 2027)  
**Concentrations:** Finance · Energy Trade | **Minor:** Entrepreneurship & Innovation

---

## Overview

This project forecasts short term U.S. energy commodity prices using publicly available data from the Federal Reserve Economic Data (FRED) database. It applies three quantitative forecasting methods to WTI crude oil and Henry Hub natural gas prices and visualizes historical trends alongside 6 month projections.

Built as an extension of commodity risk modeling work completed with the Penn State Energy Business & Finance Society, where I ranked 1st of 10 teams in a case competition modeling energy price shocks.

---

## What It Does

- Fetches live monthly price data from FRED (no API key required)
  - `DCOILWTICO` — WTI Crude Oil spot price (USD/barrel)
  - `MHHNGSP` — Henry Hub Natural Gas spot price (USD/MMBtu)
- Applies three forecasting methods across a 6-month horizon
- Outputs a summary table to the console
- Generates a two-panel chart (`forecast_chart.png`)

---

## Forecasting Methods

| Method | Description |
|---|---|
| **Simple Moving Average (SMA-12)** | Rolling 12 month mean baseline trend signal |
| **Exponential Weighted Average (EWA)** | More weight on recent prices (span=6) captures momentum |
| **Linear Trend Projection** | OLS regression on last 24 months directional bias |

Using multiple methods allows comparison of signals. When SMA and EWA converge but diverge from the linear trend, it typically indicates a mean reverting market. When all three align, it suggests stronger directional conviction relevant for hedging and trading decisions.

---

## Sample Output

```
===========================================================
  ENERGY PRICE FORECAST SUMMARY
  Horizon: 6 months from latest data point
===========================================================

  WTI Crude Oil (current: $74.20 /bbl)
  Method                  6-mo Forecast
  ------------------------------------
  SMA                      $75.10  ▲ 1.2%
  EWA                      $73.80  ▼ 0.5%
  Trend                    $71.40  ▼ 3.8%

  Henry Hub Gas (current: $2.85 /MMBtu)
  Method                  6-mo Forecast
  ------------------------------------
  SMA                       $3.10  ▲ 8.8%
  EWA                       $2.95  ▲ 3.5%
  Trend                     $3.22  ▲ 13.0%
===========================================================
```

---

## Setup & Usage

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/energy-price-forecasting.git
cd energy-price-forecasting
```

### 2. Install dependencies
```bash
pip install pandas numpy matplotlib requests
```

### 3. Run the model
```bash
python model.py
```

The script will:
1. Fetch live data from FRED (requires internet connection)
2. Print the forecast summary to your terminal
3. Save `forecast_chart.png` in the current directory
4. Display the chart

> **Offline mode:** If the FRED endpoints are unreachable, the model automatically falls back to synthetic data that mirrors real FRED structure. Swap in your own CSV exports from [fred.stlouisfed.org](https://fred.stlouisfed.org) to use real data offline.

---

## Configuration

All key parameters are at the top of `model.py`:

```python
FORECAST_MONTHS = 6    # horizon length
SMA_WINDOW      = 12   # rolling window for SMA
EWA_SPAN        = 6    # span for exponential weighted average
LOOKBACK_TREND  = 24   # months used for OLS trend fit
```

---

## Why These Methods

These three methods represent a natural progression in time series forecasting complexity:

**SMA** is the baseline it answers "where has the market been on average?" It dampens noise but lags turning points. A 12-month window captures the full seasonal cycle in natural gas markets.

**EWA** answers "where is the market going based on recent momentum?" By weighting recent observations more heavily, it responds faster to structural shifts like the 2022 energy crisis. The span parameter controls how quickly older data decays.

**Linear Trend** answers "what does the medium term directional bias look like?" Using OLS on the last 24 months isolates the slope without being overwhelmed by older structural regimes (e.g., pre-COVID price levels).

In practice, energy traders and risk analysts use combinations of these signals. This model is a foundation for more advanced approaches including ARIMA, GARCH volatility modeling, and regression against macroeconomic drivers (rig counts, storage levels, interest rates).

---

## Dependencies

| Library | Version | Purpose |
|---|---|---|
| `pandas` | ≥1.5 | Data loading, resampling, time-series manipulation |
| `numpy` | ≥1.23 | Numerical computation, OLS regression |
| `matplotlib` | ≥3.6 | Visualization |

---

## Data Sources

- **FRED (Federal Reserve Bank of St. Louis)** — [fred.stlouisfed.org](https://fred.stlouisfed.org)
  - WTI Crude Oil: [DCOILWTICO](https://fred.stlouisfed.org/series/DCOILWTICO)
  - Henry Hub Gas: [MHHNGSP](https://fred.stlouisfed.org/series/MHHNGSP)
- Data is public domain and freely available without registration

---

## Next Steps

Planned extensions to this model:

- [ ] Add ARIMA/SARIMA for seasonal decomposition
- [ ] Incorporate storage inventory data (EIA Weekly Natural Gas Storage Report)
- [ ] Add Value at Risk (VaR) calculation for a simple commodity portfolio
- [ ] Regression model: price ~ rig count + CPI + USD index
- [ ] Interactive dashboard using Streamlit

---

## Contact

David Darkwa · [LinkedIn](https://linkedin.com/in/david-darkwa) · Dkd5641@psu.edu
