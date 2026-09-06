# CAPM Analysis Report: 10 U.S. Stocks

## 1. Project Description

This project estimates the Capital Asset Pricing Model (CAPM) for 10 U.S. stocks across different industries. We estimate each stock's market beta, test for abnormal returns (alpha), and evaluate the model's assumptions using residual diagnostics, rolling beta analysis, and robustness checks across time windows and data frequencies.

**Stocks selected:** AAPL (Technology), JPM (Financials), AMZN (Consumer Discretionary), XOM (Energy), CAT (Industrials), JNJ (Healthcare), PG (Consumer Staples), NEE (Utilities), AMT (Real Estate), GOOG (Communication Services)

**Period:** January 2015 – December 2025 (2,764 trading days)

---

## 2. Theory

### CAPM (Capital Asset Pricing Model)

The CAPM states that the expected excess return of a stock is proportional to the excess return of the market, scaled by the stock's beta:

    R_i,t - R_f,t = α_i + β_i(R_m,t - R_f,t) + ε_i,t

Where:
- R_i,t = return of stock i on day t
- R_f,t = risk-free rate (3-Month Treasury bill)
- R_m,t = return of the market portfolio (S&P 500)
- α_i = alpha (abnormal return not explained by market exposure)
- β_i = beta (sensitivity to market movements)
- ε_i,t = residual (idiosyncratic, stock-specific noise)

CAPM predicts that α = 0 for all stocks. Any non-zero alpha implies the model is incomplete.

### Simple Returns vs Log Returns

- **Simple return:** (P_t - P_{t-1}) / P_{t-1}. Additive across portfolio weights. Used in this project.
- **Log return:** ln(P_t / P_{t-1}). Additive across time. Approximately equal to simple returns for small daily values.

### Excess Returns

Excess return = R_i - R_f. This isolates the compensation for bearing risk above the risk-free rate. CAPM is formulated entirely in excess return space.

### OLS (Ordinary Least Squares)

OLS finds the line Y = α + βX that minimizes the sum of squared vertical distances between observed data points and the fitted line. Under the Gauss-Markov assumptions (linearity, no multicollinearity, zero conditional mean of errors, homoscedasticity, no autocorrelation), OLS is the Best Linear Unbiased Estimator (BLUE).

### Key Regression Statistics

- **Alpha (α):** The intercept. CAPM predicts α = 0. Significant positive alpha suggests the stock outperformed its risk-adjusted benchmark.
- **Beta (β):** The slope. Measures market sensitivity. β > 1 means the stock amplifies market moves; β < 1 means it dampens them.
- **R-squared:** Fraction of the stock's excess return variance explained by the market. High R² = market-driven; low R² = idiosyncratic-driven.
- **P-value:** Probability of observing the result if the true coefficient were zero. P < 0.05 = statistically significant.
- **Durbin-Watson:** Tests autocorrelation in residuals. Values near 2.0 indicate no autocorrelation.
- **Jarque-Bera:** Tests whether residuals are normally distributed. Rejects normality when skewness ≠ 0 or kurtosis ≠ 3.

### Portfolio Risk and Diversification

Beta captures systematic (market) risk — the risk that cannot be diversified away. Idiosyncratic risk (the residual ε) can be eliminated by holding a diversified portfolio. CAPM says only systematic risk is compensated with higher expected returns.

---

## 3. Data Sources

| Data | Source | Identifier |
|------|--------|------------|
| Stock prices (adjusted close) | Yahoo Finance via yfinance | AAPL, JPM, AMZN, XOM, CAT, JNJ, PG, NEE, AMT, GOOG |
| Market index | Yahoo Finance | ^GSPC (S&P 500) |
| Risk-free rate | FRED (Federal Reserve) | DGS3MO (3-Month Treasury Bill, daily, annualized) |

---

## 4. Methodology

1. **Select stocks:** 10 stocks from 10 different GICS sectors to maximize cross-sectional variation in beta.
2. **Import prices:** Downloaded adjusted close prices for all stocks and S&P 500 (2015–2025).
3. **Clean data:** Aligned dates across all tickers, verified zero missing values, computed daily simple returns.
4. **Add risk-free rate:** Downloaded 3-Month T-bill rate from FRED, converted from annualized percentage to daily decimal (÷ 100 ÷ 252), aligned with return dates using forward-fill.
5. **Run CAPM regression:** For each stock, regressed excess stock return on excess market return using OLS (statsmodels). Extracted alpha, beta, R², t-statistics, and p-values.
6. **Evaluate results:** Produced scatter plots, residual plots, rolling 60-day beta charts, beta bar chart, Security Market Line, sub-period comparison (pre/post-COVID), and daily vs monthly frequency comparison.

---

## 5. Results Summary

### Beta and Alpha Estimates

| Stock | Sector | Beta | Alpha | Alpha p-value | R² |
|-------|--------|------|-------|---------------|-----|
| AAPL | Technology | 1.2033 | 0.0005 | 0.0482* | 0.558 |
| AMZN | Consumer Disc. | 1.1719 | 0.0006 | 0.0410* | 0.406 |
| GOOG | Communication | 1.1391 | 0.0005 | 0.0379* | 0.501 |
| JPM | Financials | 1.0961 | 0.0003 | 0.1761 | 0.523 |
| CAT | Industrials | 1.0721 | 0.0004 | 0.1338 | 0.410 |
| XOM | Energy | 0.8222 | -0.0000 | 0.9687 | 0.288 |
| AMT | Real Estate | 0.7004 | 0.0001 | 0.8335 | 0.243 |
| NEE | Utilities | 0.6415 | 0.0003 | 0.2958 | 0.217 |
| PG | Consumer Staples | 0.5124 | 0.0001 | 0.7914 | 0.246 |
| JNJ | Healthcare | 0.4927 | 0.0001 | 0.4781 | 0.235 |

*Significant at 5% level

### Beta Stability Across Time Periods

| Stock | Pre-COVID (2015–2019) | Post-COVID (2020–2025) | Change |
|-------|----------------------|----------------------|--------|
| NEE | 0.3402 | 0.7453 | +0.4051 |
| AMT | 0.5984 | 0.7357 | +0.1373 |
| JNJ | 0.6896 | 0.4250 | -0.2646 |
| CAT | 1.3309 | 0.9830 | -0.3479 |
| AMZN | 1.3204 | 1.1212 | -0.1992 |

### Daily vs Monthly Comparison

| Stock | Beta (Daily) | Beta (Monthly) | Diff |
|-------|-------------|---------------|------|
| AAPL | 1.2033 | 1.2071 | +0.004 |
| CAT | 1.0721 | 1.2733 | +0.201 |
| NEE | 0.6415 | 0.4534 | -0.188 |
| PG | 0.5124 | 0.4070 | -0.105 |

Beta rankings are preserved across frequencies. Exact values shift but qualitative conclusions hold.

---

## 6. Conclusion

### 1. Which stocks have beta above or below 1?
Five stocks have beta above 1 (aggressive): AAPL (1.20), AMZN (1.17), GOOG (1.14), JPM (1.10), and CAT (1.07). These are from cyclical or growth sectors — technology, financials, and industrials. Five stocks have beta below 1 (defensive): XOM (0.82), AMT (0.70), NEE (0.64), PG (0.51), and JNJ (0.49). These are from sectors with stable, inelastic demand.

### 2. Are defensive stocks actually lower beta?
Yes. The three lowest-beta stocks are JNJ (0.49), PG (0.51), and NEE (0.64) — healthcare, consumer staples, and utilities. Their revenues are largely independent of the business cycle, confirming CAPM's prediction that stocks with less economic sensitivity carry less systematic risk.

### 3. Which stocks have statistically significant alpha?
Three stocks have alpha significant at the 5% level: AAPL (p=0.048), AMZN (p=0.041), and GOOG (p=0.038). All are tech/growth stocks, suggesting CAPM is incomplete for companies with strong growth or momentum characteristics. The significant alphas likely reflect exposure to risk factors that CAPM omits (size, value, momentum) rather than genuine mispricing.

### 4. Does beta change during crises?
Yes, substantially. Rolling 60-day beta analysis shows betas spike during market stress — most stocks showed elevated betas during the COVID crash (March 2020). JNJ's beta occasionally went negative during flight-to-safety episodes. Sub-period analysis confirms: NEE's beta nearly doubled post-COVID (0.34 to 0.75), while JNJ became more defensive (0.69 to 0.43). This time-variation directly violates CAPM's constant-beta assumption.

### 5. Are residuals normally distributed?
No. The Jarque-Bera test rejects normality for all 10 stocks. AAPL has kurtosis of 8.43 (normal = 3), indicating fat tails — extreme daily moves occur far more frequently than a Gaussian distribution predicts. Residual plots also show volatility clustering, with variance spiking during crisis periods. OLS estimates remain unbiased, but standard errors are approximate.

### 6. Does daily vs monthly data change conclusions?
The qualitative conclusions are robust — beta rankings are preserved across both frequencies. However, exact values shift (e.g., CAT: 1.07 daily to 1.27 monthly). R-squared decreases with monthly data due to fewer observations (120 vs 2,764). The choice of data frequency affects precision but does not change economic interpretation.

### Overall Assessment
CAPM correctly identifies the direction of the risk-return tradeoff and produces economically intuitive beta rankings across sectors. However, it fails on three fronts: (1) significant alpha for growth stocks, (2) time-varying betas that violate the static model assumption, and (3) non-normal residuals with fat tails. These limitations motivate extensions including Fama-French multi-factor models, conditional CAPM, and robust estimation methods such as Newey-West standard errors.
