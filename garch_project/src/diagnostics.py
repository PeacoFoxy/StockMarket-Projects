"""Formal specification tests for volatility modelling.

The exploratory plots make volatility clustering visible. These tests
make it testable, converting a visual impression into a rejected null
hypothesis.
"""

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch


def arch_lm_test(returns, lags=(1, 5, 10, 22)):
    """Engle's (1982) LM test for ARCH effects.

    The test regresses squared residuals on their own lags:

        eps_t^2 = gamma_0 + gamma_1 eps_{t-1}^2 + ... + gamma_q eps_{t-q}^2 + u_t

    Under H0 (no ARCH effects, i.e. constant conditional variance) all
    slope coefficients are zero and T * R^2 follows a chi-squared
    distribution with q degrees of freedom.

    Rejecting H0 is the formal licence to fit GARCH. Without this step,
    the choice of a conditional variance model rests on nothing more than
    the appearance of the return plot.

    Note the test is applied to demeaned returns rather than to residuals
    from a fitted mean equation. For daily equity index data the mean is
    economically negligible relative to the variance, so the distinction
    does not materially affect the result.
    """
    rows = []
    for label, r in returns.items():
        eps = r - r.mean()                  # demean; the mean equation is a constant
        for q in lags:
            stat, pval, _, _ = het_arch(eps, nlags=q)
            # statsmodels reports only the statistic, but LM = T * R^2 by
            # construction, so R^2 recovers directly. It is the more
            # interpretable quantity: the share of variation in today's
            # squared return explained by the past q days.
            r_squared = stat / len(eps)
            # Formatted as a string so pandas renders it as a percentage.
            # R^2 is the interpretable quantity here: the share of variation
            # in today's squared return explained by the past q days.
            rows.append({
                "Index": label,
                "Lags": q,
                "LM stat": stat,
                "R^2": f"{r_squared:.1%}",
                "p-value": pval,
                "Reject H0 at 1%": pval < 0.01,
            })
    return pd.DataFrame(rows).set_index(["Index", "Lags"])


def ljung_box_test(returns, lags=(5, 10, 22)):
    """Ljung-Box test applied to returns and to squared returns.

    The contrast between the two is the entire rationale for GARCH:

      - On r_t, we expect NO significant autocorrelation. Predictable
        returns would be an arbitrage opportunity, so weak-form market
        efficiency implies this test should mostly fail to reject.

      - On r_t^2, we expect STRONG autocorrelation. Squared returns proxy
        for variance, and volatility clustering means today's magnitude
        predicts tomorrow's.

    Direction is unpredictable; magnitude is not. A model that captures
    the second without claiming to capture the first is exactly what
    GARCH is.
    """
    rows = []
    for label, r in returns.items():
        r_dm = r - r.mean()
        for q in lags:
            lb_r = acorr_ljungbox(r_dm, lags=[q], return_df=True)
            lb_r2 = acorr_ljungbox(r_dm ** 2, lags=[q], return_df=True)
            rows.append({
                "Index": label,
                "Lags": q,
                "Q(r)": lb_r["lb_stat"].iloc[0],
                "p(r)": lb_r["lb_pvalue"].iloc[0],
                "Q(r^2)": lb_r2["lb_stat"].iloc[0],
                "p(r^2)": lb_r2["lb_pvalue"].iloc[0],
            })
    return pd.DataFrame(rows).set_index(["Index", "Lags"])
