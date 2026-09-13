# GARCH Models for Global Equity Indices

Angelina Huang  
September 2026

---

## 1. Project description

This project estimates conditional volatility models for four major equity indices and compares volatility persistence across markets. Beyond estimation, it tests whether the modelling choices survive out-of-sample validation, using 99% Value-at-Risk backtests as the criterion.

| Index | Ticker | Region |
|---|---|---|
| S&P 500 | `^GSPC` | United States |
| EURO STOXX 50 | `^STOXX50E` | Eurozone |
| Nikkei 225 | `^N225` | Japan |
| FTSE 100 | `^FTSE` | United Kingdom |

**Sample:** 2 April 2007 – 11 September 2026, approximately 4,800 observations per index. The start date is set by Yahoo Finance's history for `^STOXX50E`; all four indices are restricted to a common start so that estimates remain comparable.

---

## 2. Theory

### Volatility clustering

Daily equity returns are close to serially uncorrelated but far from independent. Large moves cluster: a turbulent day is disproportionately likely to be followed by another. Any model assuming constant variance misstates risk in both directions — too conservative in calm periods, badly too permissive in crises.

### GARCH(1,1)

Bollerslev (1986) specifies the conditional variance as

$$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$

Substituting the recursion into itself gives

$$\sigma_t^2 = \frac{\omega}{1-\beta} + \alpha\sum_{i=1}^{\infty}\beta^{i-1}\epsilon_{t-i}^2$$

so GARCH(1,1) is an ARCH($\infty$) with geometrically decaying weights. Two parameters generate an infinite sequence — the reason it is preferred to ARCH(q), which would require roughly twenty-two lags to capture the dependence observed here.

| Parameter | Interpretation |
|---|---|
| $\alpha$ | Reaction to yesterday's shock |
| $\beta$ | Memory of yesterday's variance |
| $\alpha+\beta$ | Persistence; stationarity requires $< 1$ |
| $\ln(0.5)/\ln(\alpha+\beta)$ | Half-life of a variance shock, in trading days |
| $\omega/(1-\alpha-\beta)$ | Implied unconditional variance |

### Asymmetry

GJR-GARCH (Glosten, Jagannathan and Runkle, 1993) adds an indicator term:

$$\sigma_t^2 = \omega + \alpha\epsilon_{t-1}^2 + \gamma\epsilon_{t-1}^2\mathbb{1}_{\{\epsilon_{t-1}<0\}} + \beta\sigma_{t-1}^2$$

Negative shocks receive $\alpha+\gamma$, positive shocks $\alpha$. Persistence becomes $\alpha + \gamma/2 + \beta$, since the indicator fires on half of observations in expectation.

The economic mechanism is contested. Black (1976) attributes it to leverage: a price decline raises the debt-to-equity ratio and hence firm risk. Volatility feedback offers the reverse causality — rising volatility raises required returns, depressing prices immediately.

### Distributional assumptions

GARCH generates fat unconditional tails through variance mixing even with Gaussian innovations. Whether it generates *enough* is an empirical question, answered by examining the standardised residuals $\hat z_t = \hat\epsilon_t/\hat\sigma_t$. Two extensions are used here: Student-$t$ (one parameter, $\nu$, controlling tail thickness) and Hansen's (1994) skewed Student-$t$ (adding $\lambda$, allowing the two tails to differ).

---

## 3. Data

| Item | Source | Detail |
|---|---|---|
| Index prices | Yahoo Finance via `yfinance` | Daily close, `auto_adjust=True` |
| Returns | Computed | $100 \times \ln(P_t/P_{t-1})$ |

**Log returns** are used because they are time-additive, making variance aggregation and the square-root-of-time rule exact, and because they are supported on the whole real line, consistent with the Gaussian or Student-$t$ innovations the model assumes.

**Scaling by 100** is a numerical requirement. Unscaled daily returns have variance around $10^{-4}$, which degrades the accuracy of the gradient and Hessian used by the MLE optimiser and can prevent convergence.

**No calendar alignment.** The four markets trade on different calendars. Merging them and dropping incomplete rows would discard 6.5–9.5% of observations, and would do so systematically rather than at random: holidays cluster, so alignment removes contiguous blocks and severs the variance recursion across each gap. Each index is therefore kept as an independent series.

---

## 4. Methodology

1. **Establish the need for a conditional variance model.** ARCH-LM (Engle 1982) and Ljung–Box tests on raw and squared returns.
2. **Estimate GARCH(1,1)** by maximum likelihood with Gaussian innovations. Extract persistence, half-life and standard errors.
3. **Test the distributional assumption.** Examine standardised residuals; refit with Student-$t$ if excess kurtosis remains.
4. **Test symmetry.** Residual skewness under symmetric models motivates GJR and skewed-$t$.
5. **Compare specifications** by AIC and nested likelihood ratio tests.
6. **Check robustness** to sample period and examine cross-market differences.
7. **Validate out-of-sample** via 99% VaR backtests, using Kupiec (1995) for coverage and Christoffersen (1998) for independence.

Each stage diagnoses a specific failure of the previous specification before introducing the fix.

---

## 5. Results

### 5.1 Stylised facts

| Index | Ann. vol (%) | Skewness | Excess kurtosis |
|---|---|---|---|
| SPX | 19.78 | −0.469 | 12.47 |
| SX5E | 21.78 | −0.304 | 7.71 |
| Nikkei | 23.95 | −0.444 | 8.21 |
| FTSE | 17.83 | −0.429 | 10.37 |

All four series reject normality (Jarque–Bera $p < 10^{-300}$). SPX has the highest kurtosis despite the lowest volatility — calm on average, with the most concentrated tail risk. Its sample minimum of −12.77% is approximately a 10-sigma event under normality.

### 5.2 Specification tests

ARCH-LM rejects the null of constant conditional variance in all sixteen tests (four indices × four lag lengths), with $p$-values indistinguishable from zero. At twenty-two lags the auxiliary regression $R^2$ reaches 30% for SPX: the past month of squared returns explains nearly a third of today's.

Ljung–Box on squared returns exceeds the same statistic on returns by a factor of 40 to 81. Returns do show mild autocorrelation ($|\bar\rho| \approx 0.04$), attributable largely to non-synchronous trading in index constituents, but the magnitude is economically negligible and a constant mean is retained throughout.

### 5.3 Final model estimates

GJR-GARCH(1,1) with skewed Student-$t$ innovations:

| Index | $\alpha$ | $\gamma$ | $\beta$ | Persistence | Half-life | $\nu$ | $\lambda$ |
|---|---|---|---|---|---|---|---|
| SPX | 0.000 | 0.250 | 0.859 | 0.9843 | 43.8 | 6.61 | −0.169 |
| SX5E | 0.000 | 0.226 | 0.869 | 0.9823 | 38.9 | 6.61 | −0.114 |
| Nikkei | 0.034 | 0.172 | 0.843 | 0.9623 | 18.0 | 7.78 | −0.088 |
| FTSE | 0.000 | 0.220 | 0.864 | 0.9743 | 26.6 | 7.12 | −0.126 |

### 5.4 Model comparison

AIC, with the likelihood ratio for each nested step:

| Index | Normal | $t$ | GJR-$t$ | GJR-skew$t$ |
|---|---|---|---|---|
| SPX | 13,380 | 13,098 | 12,923 | **12,852** |
| SX5E | 15,298 | 15,027 | 14,832 | **14,801** |
| Nikkei | 16,006 | 15,829 | 15,740 | **15,721** |
| FTSE | 13,034 | 12,800 | 12,661 | **12,620** |

Every step improves AIC by between 19 and 283. All likelihood ratios exceed the $\chi^2_1$ critical value of 6.63 by one to two orders of magnitude. Gains diminish in order: tail thickness matters most, then the variance response to sign, then tail asymmetry.

### 5.5 Out-of-sample VaR

Walk-forward from January 2016, refitting annually, 99% one-day VaR:

| Specification | Mean violation rate | Kupiec failures | Christoffersen failures |
|---|---|---|---|
| Normal GARCH | 2.02% | 4 of 4 | 0 of 4 |
| $t$-GARCH | 1.31% | 2 of 4 | 1 of 4 |
| GJR-skew$t$ | 1.17% | 0 of 4 | 0 of 4 |

---

## 6. Conclusions

### Which index has the highest volatility?

The Nikkei, at 23.9% annualised, followed by SX5E (21.8%), SPX (19.8%) and FTSE (17.8%). The rolling volatility panels show this is not driven by crises: the Nikkei sits above the others through 2013–2016 and again from 2024, periods in which the other three are calm. These coincide with Abenomics-era monetary expansion and the 2024 exit from negative rates. Japan's elevated volatility reflects a domestic policy regime as much as exposure to global shocks.

### Which index has the most persistent volatility?

By point estimate, SPX: a variance half-life of 44 trading days against 18 for the Nikkei.

**This ordering is not statistically established.** The SPX–Nikkei comparison gives $t = 0.91$, $p \approx 0.36$; the 95% confidence intervals overlap substantially. The ranking is invariant across all four specifications tested, which makes sampling noise an unlikely explanation, but it is not invariant to the sample period — FTSE moves from second to last across a 2016 split. Only SPX's position is robust on both dimensions.

What the data does establish is that persistence is confined to a narrow band across four markets with different regulatory regimes, investor compositions and trading hours. Volatility clustering appears to be a near-universal property of equity index returns.

**Level and persistence rank in opposite orders.** The most volatile market has the shortest memory; the least volatile has the longest. These are distinct properties, and the distinction has practical content: Nikkei volatility offers higher premia with faster mean reversion, while SPX volatility shocks require materially longer hedging horizons.

### Do crisis periods show clear volatility spikes?

Yes, and the *shape* of the spike differs by episode. The 2008 crisis produces a plateau — volatility between 40% and 60% for roughly nine months. The 2020 crash produces a spike: a higher peak (97% for SPX against 85% in 2008) but back near 20% within months. The 2011 European sovereign debt episode is regional, raising SX5E to 50% while the Nikkei barely responds.

### Does Student-$t$ fit better than normal errors?

Decisively. AIC improves by 177 to 283 for one parameter, with likelihood ratios around 570. But the in-sample evidence understates the case: out-of-sample, normal innovations breach 99% VaR at roughly twice the nominal rate and are rejected by Kupiec in all four markets. Under a Basel traffic-light framework that is red-zone performance.

A caveat on interpretation. The raw excess kurtosis of 7.7–12.5 does *not* by itself justify Student-$t$, because GARCH generates fat unconditional tails through variance mixing regardless of the innovation distribution. The case rests on the standardised residuals, which retain excess kurtosis of 1.4 to 2.0 after fitting under normality.

### Are standardised residuals still autocorrelated?

No. Ljung–Box on squared standardised residuals falls from 2,770–6,824 on raw squared returns to 19.8–30.7, a reduction exceeding 99%, and eleven of twelve tests fail to reject white noise. The variance equation is well specified.

Two traces remain. The Nikkei is marginally significant at five and ten lags but not at twenty-two, consistent with the Japan-specific regimes identified above. SPX retains mild autocorrelation in the residuals themselves — a mean-equation artefact, not a variance problem.

### How different are volatility dynamics across regions?

Less different than expected in persistence, more different than expected in asymmetry.

The estimated ARCH coefficient is **zero** for SPX, SX5E and FTSE: positive shocks move conditional variance by nothing at all. The entire response runs through $\gamma$. The Nikkei is the only market with a non-zero response to good news, and even there the asymmetry ratio is roughly six to one.

This is the fifth independent indicator distinguishing Japan from the other three — it also has the lowest raw kurtosis, the smallest skewness, $\lambda$ closest to zero, and the smallest gain from each model refinement. Plausibly its volatility is driven more by currency and monetary-policy shocks, which are two-sided, than by the credit and leverage channels dominant elsewhere.

### Overall assessment

The exercise supports three conclusions. Conditional volatility models are necessary, not optional — the ARCH-LM evidence is overwhelming and the VaR backtests show the cost of ignoring it. The distributional and asymmetry extensions earn their parameters both in-sample and out-of-sample. And cross-market differences in persistence, while consistently ordered, are smaller than the shared structure: four very different markets produce remarkably similar volatility dynamics.

---

## 7. Limitations

**GARCH cannot distinguish the nature of a shock.** The Nikkei's −12% move on 5 August 2024 was a yen carry-trade unwind — technical, position-driven, resolved within weeks. Comparable moves in 2008 reflected deteriorating credit fundamentals and persisted for a year. The model observes only $\epsilon_t^2$ and propagates both identically. Regime-switching or jump-diffusion specifications address this at substantial cost in complexity.

**Parameters are assumed constant over nineteen years.** The sub-sample results reject this: persistence falls in every market after 2016 while $\gamma$ rises. Recent markets are more reactive to bad news and faster to recover — plausibly reflecting passive flows, algorithmic execution and mechanical deleveraging. A single parameter set describes an average of at least two regimes.

**$\hat\alpha = 0$ sits on a constrained boundary** in three of four markets, where standard asymptotics do not apply exactly. The reported standard errors for $\alpha$ are approximations.

**The mean equation is a constant** despite statistically significant return autocorrelation. The magnitude is economically negligible, but the residual diagnostics confirm it is not zero.

**Persistence differences are not significant.** Reported rankings should be read as point estimates with overlapping intervals, not as established market characteristics.

---

## References

Black, F. (1976). Studies of stock price volatility changes.  
Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. *Journal of Econometrics*, 31(3).  
Christoffersen, P. (1998). Evaluating interval forecasts. *International Economic Review*, 39(4).  
Engle, R. (1982). Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation. *Econometrica*, 50(4).  
Engle, R. and Ng, V. (1993). Measuring and testing the impact of news on volatility. *Journal of Finance*, 48(5).  
Glosten, L., Jagannathan, R. and Runkle, D. (1993). On the relation between the expected value and the volatility of the nominal excess return on stocks. *Journal of Finance*, 48(5).  
Hansen, B. (1994). Autoregressive conditional density estimation. *International Economic Review*, 35(3).  
Kupiec, P. (1995). Techniques for verifying the accuracy of risk measurement models. *Journal of Derivatives*, 3(2).  
Nelson, D. (1991). Conditional heteroskedasticity in asset returns: a new approach. *Econometrica*, 59(2).