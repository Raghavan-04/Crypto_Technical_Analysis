
# 📊 FOCUS: Institutional Options Flow Analyzer

**FOCUS** is a high-frequency trading dashboard designed to visualize institutional options flow, Greek exposure, and market sentiment in real-time. It bridges the gap between raw options chain data and actionable trading signals by calculating metrics like **Max Pain**, **Gamma Exposure**, and **Net Delta** on the fly.

## 🚀 Key Features

* **Real-Time Dashboard:** Auto-updating 8-grid visualization using `Matplotlib`.
* **Greeks Analysis:** Tracks **Net Delta** (Directional Bias) and **Total Gamma** (Volatility Sensitivity/Pinning).
* **Max Pain Theory:** Visualizes the "Max Pain" price magnet vs. current Spot Price.
* **Smart Money Tracking:** Monitors significant Open Interest (OI) changes and "Hot Strikes."
* **Dynamic Alerts:** Audio (Windows) and visual alerts for High Gamma spikes, PCR shifts, and Delta inversions.
* **Risk Scoring:** Automated calculation of market risk levels (Low/Med/High).
* **Data Export:** One-key CSV export for offline analysis.

## 📦 Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/focus-analyzer.git
cd focus-analyzer

```


2. **Install dependencies:**
```bash
pip install numpy pandas matplotlib

```


*(Note: `winsound` is used for audio on Windows. Linux/Mac users will receive visual terminal alerts only).*

## ⚙️ Data Configuration (Critical)

The script monitors a local file named `market_data.json`. You must have an external scraper or data provider writing to this file. The structure **must** match the following schema:

```json
{
  "metadata": {
    "underlying": "SPY",
    "spot_price": 450.50,
    "max_pain": 452.00,
    "timestamp": "2023-10-27 14:30:00"
  },
  "options_chain": [
    {
      "strike": 450,
      "type": "Call",
      "oi": 15000,
      "volume": 5000,
      "delta": 0.52,
      "gamma": 0.04,
      "vega": 0.12,
      "theta": -0.05,
      "mark_price": 2.50
    },
    {
      "strike": 450,
      "type": "Put",
      "oi": 12000,
      "volume": 3000,
      "delta": -0.48,
      "gamma": 0.04,
      "vega": 0.12,
      "theta": -0.04,
      "mark_price": 2.10
    }
  ]
}

```

## 🖥️ Usage

1. Ensure your data feed is updating `market_data.json`.
2. Run the analyzer:
```bash
python main.py

```



### ⌨️ Keyboard Controls

Click on the plot window to focus it, then use these keys:

* **`E`**: **Export** current analysis to timestamped CSV files (`strike_analysis_*.csv`, `market_summary_*.csv`).
* **`R`**: **Reset** history graphs and alert logs.
* **`Q`**: **Quit** the application safely.

## 📊 Dashboard Modules

The GUI is divided into 8 strategic panels:

1. **Price Action:** Spot price vs. Max Pain with calculated Support/Resistance levels.
2. **Sentiment Gauge:** Visual needle showing Bullish/Bearish bias based on PCR and Delta.
3. **OI Imbalance:** Horizontal bars showing Net OI (Calls vs. Puts) per strike.
4. **Gamma Wall:** Identifies "Sticky Strikes" where dealers are heavily hedged.
5. **Delta Profile:** Shows the net directional exposure of market makers.
6. **PCR Trend:** Historical tracking of Put/Call Ratio over the session.
7. **Volume Heatmap:** Visualizes where the most trading activity is occurring right now.
8. **Signal Summary:** Text-based aggregation of Bull/Bear signals and Trade Recommendations.

## ⚠️ Disclaimer

This software is for **educational and research purposes only**. It does not constitute financial advice. Option Greeks and Open Interest data are lagging indicators and should not be the sole basis for real-money trading decisions.
