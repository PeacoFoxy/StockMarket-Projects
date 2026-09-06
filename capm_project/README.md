# CAPM Analysis: 10 U.S. Stocks

**Path:** `~/Desktop/Student projects/capm_project/`

Estimating market beta for 10 U.S. stocks across different GICS sectors using the Capital Asset Pricing Model (CAPM). The project tests whether beta correctly captures systematic risk and evaluates CAPM's assumptions through residual diagnostics, rolling beta analysis, and robustness checks.

## Stocks
AAPL, JPM, AMZN, XOM, CAT, JNJ, PG, NEE, AMT, GOOG

## Period
January 2015 – December 2025 (2,764 trading days)

## Folder Structure
```
capm_project/
├── README.md
├── requirements.txt
├── data/
│   ├── stock_prices.csv
│   ├── market_prices.csv
│   └── risk_free_rate.csv
├── notebooks/
│   └── capm_analysis.ipynb
├── outputs/
│   ├── scatter_plots.png
│   ├── residual_plots.png
│   ├── rolling_beta.png
│   ├── beta_comparison.png
│   ├── sml.png
│   ├── capm_results.csv
│   ├── regression_summary.csv
│   ├── daily_vs_monthly.csv
│   └── time_window_comparison.csv
├── reports/
│   └── CAPM_Report.md
└── src/
    ├── data_loader.py
    ├── regression.py
    └── plots.py
```

## Key Findings
- Betas range from 0.49 (JNJ) to 1.20 (AAPL), matching economic intuition across sectors
- 3 stocks (AAPL, AMZN, GOOG) show statistically significant alpha, suggesting CAPM is incomplete for growth stocks
- Rolling beta analysis reveals betas are time-varying and spike during crises, violating CAPM's constant-beta assumption
- Residuals exhibit fat tails (kurtosis up to 8.4) and volatility clustering
- Results are robust to daily vs monthly data frequency

## Setup
```
pip install -r requirements.txt
```
