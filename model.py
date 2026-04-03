"""
Energy Price Forecasting Model
================================
Author : David Darkwa
Major  : Energy Business Finance, Penn State (Expected May 2027)
Date   : March 2026

Description
-----------
This model fetches publicly available WTI crude oil and Henry Hub natural
gas price data from FRED (Federal Reserve Economic Data), applies three
forecasting methods, and visualizes historical trends alongside projections.

Forecasting Methods
-------------------
1. Simple Moving Average (SMA)   — rolling 12-month mean
2. Exponential Weighted Average (EWA) — more weight on recent prices
3. Linear Trend Projection       — OLS regression on last 24 months

Data Source
-----------
FRED (Federal Reserve Bank of St. Louis)
  WTI Crude Oil : DCOILWTICO  (daily -> resampled monthly)
  Henry Hub Gas : MHHNGSP     (monthly spot price, $/MMBtu)

No API key required for FRED CSV downloads.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
FORECAST_MONTHS   = 6       # how many months ahead to project
SMA_WINDOW        = 12      # rolling window for SMA (months)
EWA_SPAN          = 6       # span for exponential weighted average
LOOKBACK_TREND    = 24      # months used to fit the linear trend

WTI_URL  = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DCOILWTICO"
GAS_URL  = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MHHNGSP"

COLORS = {
    "historical" : "#1a2744",
    "sma"        : "#0F6E56",
    "ewa"        : "#BA7517",
    "trend"      : "#993C1D",
    "forecast_bg": "#F1EFE8",
    "grid"       : "#D3D1C7",
}

# ── Data Loading ──────────────────────────────────────────────────────────────

def load_fred_series(url: str, label: str) -> pd.Series:
    """Download a FRED CSV series, parse dates, return a monthly Series."""
    print(f"  Fetching {label} from FRED...")
    df = pd.read_csv(url, parse_dates=["DATE"], index_col="DATE", na_values=".")
    series = df.iloc[:, 0].dropna()
    # Resample to monthly average (handles daily series like WTI)
    series = series.resample("MS").mean()
    print(f"  {label}: {len(series)} monthly observations "
          f"({series.index[0].strftime('%b %Y')} – {series.index[-1].strftime('%b %Y')})")
    return series


def load_data() -> dict:
    """Load both price series. Falls back to realistic synthetic data if offline."""
    try:
        wti = load_fred_series(WTI_URL, "WTI Crude Oil ($/bbl)")
        gas = load_fred_series(GAS_URL, "Henry Hub Gas ($/MMBtu)")
        return {"wti": wti, "gas": gas}
    except Exception as e:
        print(f"\n  [!] Network unavailable ({e}).")
        print("      Generating synthetic data that mirrors FRED structure.")
        print("      Replace with real FRED data by running when online.\n")
        return generate_synthetic_data()


def generate_synthetic_data() -> dict:
    """
    Produces realistic synthetic energy price series for offline use.
    Structure mirrors actual FRED data exactly — swap in the real fetch
    once you have internet access.
    """
    rng   = np.random.default_rng(42)
    dates = pd.date_range("2015-01-01", periods=134, freq="MS")  # ~11 years

    # WTI: mean-reverting around $70, with COVID crash + recovery baked in
    wti_base = np.array([
        50, 48, 46, 44, 45, 48, 47, 45, 43, 42, 40, 38,
        38, 40, 43, 46, 48, 50, 51, 49, 47, 48, 50, 52,
        54, 56, 59, 62, 64, 63, 65, 64, 62, 60, 58, 57,
        56, 55, 57, 58, 60, 62, 63, 64, 65, 63, 61, 60,
        59, 58, 30, 20, 18, 22, 28, 35, 40, 42, 45, 48,   # COVID crash
        50, 54, 58, 63, 68, 72, 78, 85, 95, 105, 110, 105, # recovery + surge
        98, 92, 88, 85, 82, 80, 78, 76, 75, 74, 73, 72,
        73, 74, 76, 78, 79, 80, 78, 76, 74, 73, 72, 71,
        72, 73, 74, 76, 77, 78, 79, 80, 79, 78, 77, 76,
        75, 74, 73, 74, 75, 76, 77, 78, 79, 80, 79, 78,
        77, 76, 75, 74, 73, 72, 71, 70, 71, 72, 73, 74,
        75, 76,
    ], dtype=float)
    noise = rng.normal(0, 1.8, len(dates))
    wti   = pd.Series(wti_base[:len(dates)] + noise, index=dates, name="WTI")

    # Henry Hub: seasonally driven, mean around $3.50
    t       = np.arange(len(dates))
    season  = 0.8 * np.sin(2 * np.pi * t / 12 + np.pi)   # winter peaks
    trend   = 0.004 * t
    hh_base = 3.2 + trend + season + rng.normal(0, 0.25, len(dates))
    # 2022 energy crisis spike
    hh_base[84:96] += np.array([1, 2, 3, 4, 4.5, 4, 3, 2.5, 2, 1.5, 1, 0.5])
    gas = pd.Series(hh_base, index=dates, name="HenryHub")

    return {"wti": wti, "gas": gas}


# ── Forecasting ───────────────────────────────────────────────────────────────

def forecast_series(series: pd.Series, n_months: int) -> pd.DataFrame:
    """
    Apply three forecasting methods to a price series.
    Returns a DataFrame with historical values + n_months of projections.
    """
    s = series.copy().dropna()

    # 1. Simple Moving Average
    sma_last = s.rolling(SMA_WINDOW).mean().iloc[-1]
    sma_proj = np.full(n_months, sma_last)

    # 2. Exponential Weighted Average
    ewa_last = s.ewm(span=EWA_SPAN).mean().iloc[-1]
    # EWA decays toward SMA over the horizon
    alpha    = 2 / (EWA_SPAN + 1)
    ewa_proj = ewa_last + (sma_last - ewa_last) * (
        1 - (1 - alpha) ** np.arange(1, n_months + 1)
    )

    # 3. Linear Trend (OLS on last LOOKBACK_TREND months)
    recent  = s.iloc[-LOOKBACK_TREND:]
    x       = np.arange(len(recent))
    slope, intercept = np.polyfit(x, recent.values, 1)
    x_proj  = np.arange(len(recent), len(recent) + n_months)
    trend_proj = slope * x_proj + intercept

    # Build forecast index
    last_date  = s.index[-1]
    proj_dates = pd.date_range(
        last_date + pd.DateOffset(months=1), periods=n_months, freq="MS"
    )

    proj_df = pd.DataFrame({
        "SMA"   : sma_proj,
        "EWA"   : ewa_proj,
        "Trend" : trend_proj,
    }, index=proj_dates)

    return s, proj_df


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_commodity(ax, series, proj_df, title, ylabel, unit):
    """Plot one commodity panel: historical + three forecast bands."""
    display = series.iloc[-48:]   # last 4 years for readability

    # Shade forecast zone
    ax.axvspan(proj_df.index[0], proj_df.index[-1],
               color=COLORS["forecast_bg"], alpha=0.6, zorder=0)

    # Historical
    ax.plot(display.index, display.values,
            color=COLORS["historical"], linewidth=1.8,
            label="Historical price", zorder=3)

    # SMA (flat projection)
    ax.plot(proj_df.index, proj_df["SMA"],
            color=COLORS["sma"], linewidth=1.5,
            linestyle="--", label=f"SMA-{SMA_WINDOW}mo", zorder=4)

    # EWA
    ax.plot(proj_df.index, proj_df["EWA"],
            color=COLORS["ewa"], linewidth=1.5,
            linestyle="-.", label=f"EWA (span={EWA_SPAN})", zorder=4)

    # Trend
    ax.plot(proj_df.index, proj_df["Trend"],
            color=COLORS["trend"], linewidth=1.5,
            linestyle=":", label="Linear trend", zorder=4)

    # Vertical separator
    ax.axvline(proj_df.index[0], color=COLORS["grid"],
               linewidth=0.8, linestyle="-")

    # Annotation: current price
    last_val  = series.iloc[-1]
    last_date = series.index[-1]
    ax.annotate(f"  {unit}{last_val:.1f}",
                xy=(last_date, last_val),
                fontsize=8, color=COLORS["historical"], va="center")

    # Axes formatting
    ax.set_title(title, fontsize=11, fontweight="bold",
                 color=COLORS["historical"], pad=8)
    ax.set_ylabel(ylabel, fontsize=9, color="#444441")
    ax.tick_params(labelsize=8, colors="#444441")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.yaxis.grid(True, color=COLORS["grid"], linewidth=0.5, linestyle="--")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(COLORS["grid"])
    ax.legend(fontsize=8, loc="upper left", framealpha=0.8,
              edgecolor=COLORS["grid"])

    # Label forecast zone
    mid_date = proj_df.index[len(proj_df) // 2]
    ax.text(mid_date, ax.get_ylim()[0] * 1.01,
            "← Forecast →", ha="center", fontsize=7.5,
            color="#888780", style="italic")


def build_figure(data: dict):
    """Compose the full two-panel figure."""
    wti_s, wti_proj = forecast_series(data["wti"], FORECAST_MONTHS)
    gas_s, gas_proj = forecast_series(data["gas"], FORECAST_MONTHS)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), dpi=150)
    fig.patch.set_facecolor("white")

    plot_commodity(ax1, wti_s, wti_proj,
                   "WTI Crude Oil — Price Forecast",
                   "Price (USD / barrel)", "$")
    plot_commodity(ax2, gas_s, gas_proj,
                   "Henry Hub Natural Gas — Price Forecast",
                   "Price (USD / MMBtu)", "$")

    # Figure header
    fig.suptitle(
        "U.S. Energy Price Forecasting Model",
        fontsize=14, fontweight="bold", color=COLORS["historical"], y=0.98
    )
    fig.text(
        0.5, 0.955,
        f"David Darkwa · Penn State EBF · "
        f"Data: FRED (DCOILWTICO, MHHNGSP) · "
        f"Generated {datetime.now().strftime('%B %d, %Y')}",
        ha="center", fontsize=8, color="#888780"
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig("forecast_chart.png", bbox_inches="tight", dpi=150)
    print("  Chart saved → forecast_chart.png")
    plt.show()


# ── Summary Table ─────────────────────────────────────────────────────────────

def print_summary(data: dict):
    """Print a clean forecast summary to console."""
    print("\n" + "="*58)
    print("  ENERGY PRICE FORECAST SUMMARY")
    print(f"  Horizon: {FORECAST_MONTHS} months from latest data point")
    print("="*58)

    for key, label, unit in [
        ("wti", "WTI Crude Oil", "$/bbl"),
        ("gas", "Henry Hub Gas", "$/MMBtu"),
    ]:
        s, proj = forecast_series(data[key], FORECAST_MONTHS)
        current = s.iloc[-1]
        print(f"\n  {label} (current: {unit[0]}{current:.2f} {unit})")
        print(f"  {'Method':<22} {'6-mo Forecast':>14}")
        print(f"  {'-'*36}")
        for method in ["SMA", "EWA", "Trend"]:
            val = proj[method].iloc[-1]
            chg = (val - current) / current * 100
            arrow = "▲" if chg >= 0 else "▼"
            print(f"  {method:<22} {unit[0]}{val:>7.2f}  {arrow} {abs(chg):.1f}%")

    print("\n" + "="*58)
    print("  Methods:")
    print(f"  SMA   — {SMA_WINDOW}-month Simple Moving Average")
    print(f"  EWA   — Exponential Weighted Avg (span={EWA_SPAN})")
    print(f"  Trend — OLS linear regression on last {LOOKBACK_TREND} months")
    print("="*58 + "\n")


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nU.S. Energy Price Forecasting Model")
    print("David Darkwa · Penn State EBF · March 2026\n")
    print("Loading data...")
    data = load_data()
    print_summary(data)
    print("Building chart...")
    build_figure(data)
