# GARCH Models for Global Equity Indices

Conditional volatility models for four major equity indices, with out-of-sample VaR validation.

**Indices:** S&P 500 (`^GSPC`), EURO STOXX 50 (`^STOXX50E`), Nikkei 225 (`^N225`), FTSE 100 (`^FTSE`)  
**Period:** April 2007 – September 2026 (~4,800 observations per index)

## Key findings

**Volatility level and volatility persistence rank in opposite orders.** The Nikkei is the most volatile market unconditionally (23.9% annualised) but has the shortest memory (half-life 18 days). SPX is the least volatile (19.8%) but the most persistent (44 days).

**Good news moves variance by nothing.** In three of four markets the estimated ARCH coefficient is zero: only negative shocks raise conditional variance. The Nikkei alone responds to positive shocks, and even there the asymmetry ratio is 6:1.

**Persistence differences are not statistically established.** The SPX–Nikkei comparison gives t = 0.91. The ranking is invariant to model specification but not to sample period — FTSE moves from second to last across a 2016 split.

**Model choice has material consequences for risk measurement.** Out-of-sample at 99% confidence:

| Specification | Violation rate | Backtests failed |
|---|---|---|
| Normal GARCH | 2.02% | 4 of 8 |
| t-GARCH | 1.31% | 3 of 8 |
| GJR-skewt | 1.17% | 0 of 8 |

Normal innovations breach at twice the nominal rate — red-zone territory under a Basel traffic-light framework.

## Method

Seven stages, each diagnosing a specific failure before fixing it:

1. **Data** — log returns, summary statistics. No calendar alignment across markets.
2. **Specification tests** — ARCH-LM rejects constant variance in all 16 tests; Ljung-Box on squared returns exceeds that on returns by 40–80×.
3. **GARCH(1,1)** — persistence, half-life, standard errors.
4. **Student-t** — standardised residuals retain excess kurtosis of 1.4–2.0 under normality.
5. **GJR-skewt** — residual skewness of −0.37 to −0.62 is untouched by symmetric models.
6. **Cross-market comparison** — sub-sample robustness, news impact curves.
7. **VaR backtesting** — Kupiec and Christoffersen, walk-forward from 2016.

## Structure

    garch_project/
    ├── data/               cached return series
    ├── notebooks/          garch_analysis.ipynb
    ├── src/
    │   ├── data_loader.py      download, log returns, summary stats
    │   ├── diagnostics.py      ARCH-LM, Ljung-Box
    │   ├── garch_models.py     estimation, persistence, half-life
    │   ├── var_backtest.py     walk-forward VaR, Kupiec, Christoffersen
    │   └── plots.py            all figures
    ├── outputs/            figures and result tables
    ├── reports/            written report
    └── requirements.txt

## Running it

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    jupyter lab notebooks/garch_analysis.ipynb

The notebook runs top to bottom. Data is cached in `data/index_returns.csv`; Yahoo Finance revises history over time, so the cached snapshot is what reproduces the reported figures exactly.

## Limitations

GARCH cannot distinguish the nature of a shock — the Nikkei's −12% yen carry unwind in August 2024 and comparable moves during the 2008 credit crisis enter the model identically. Parameters are held constant across nineteen years, which the sub-sample results reject. Persistence differences across markets are not statistically significant.
