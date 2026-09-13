"""Out-of-sample VaR forecasting and backtesting."""

import numpy as np
import pandas as pd
from scipy import stats
from arch import arch_model


def rolling_var_forecast(series, split, alpha=0.01, dist="skewt",
                         p=1, o=1, q=1, refit_every=250):
    """One-step-ahead VaR forecasts on a walk-forward basis.

    Refitting daily would be exact but slow; every 250 days (one trading
    year) is the usual compromise. Between refits the parameters are held
    fixed while sigma_t keeps updating from new data, so forecasts remain
    genuinely out-of-sample.
    """
    dates = series.index
    split_pos = dates.searchsorted(pd.Timestamp(split))

    var_forecasts, realised, used_dates = [], [], []
    res = None

    for i in range(split_pos, len(series)):
        # Refit on all data up to (but not including) day i
        if res is None or (i - split_pos) % refit_every == 0:
            model = arch_model(series.iloc[:i], mean="Constant",
                               vol="GARCH", p=p, o=o, q=q, dist=dist)
            res = model.fit(disp="off")

        # Compute sigma_t directly from the GJR recursion rather than via
        # res.forecast(). The fitted result object stores sigma only up to
        # its own estimation sample; here we need it rolled forward with
        # fixed parameters as new observations arrive between refits.
        #
        #   sigma^2_t = omega + alpha*eps^2_{t-1}
        #               + gamma*eps^2_{t-1}*1{eps_{t-1}<0} + beta*sigma^2_{t-1}
        mu = res.params["mu"]
        om = res.params["omega"]
        al = res.params["alpha[1]"]
        ga = res.params.get("gamma[1]", 0.0)
        be = res.params["beta[1]"]

        eps_hist = (series.iloc[:i] - mu).values
        # Initialise at the unconditional variance, then iterate forward.
        # Any reasonable starting value washes out within a few dozen steps.
        persist = al + ga / 2 + be
        s2 = om / (1 - persist)
        for e in eps_hist:
            s2 = om + al * e**2 + ga * e**2 * (e < 0) + be * s2
        sigma = np.sqrt(s2)

        # Left-tail quantile of the standardised innovation distribution.
        # Shape parameters are whatever the chosen dist requires: none for
        # normal, eta for t, (eta, lambda) for skewt.
        shape = [res.params[nm] for nm in res.params.index
                 if nm in ("nu", "eta", "lambda")]
        q_alpha = res.model.distribution.ppf(alpha, shape) if shape \
            else stats.norm.ppf(alpha)
        q_alpha = float(np.asarray(q_alpha).ravel()[0])

        var_forecasts.append(-(mu + sigma * q_alpha))
        realised.append(series.iloc[i])
        used_dates.append(dates[i])

    out = pd.DataFrame({"VaR": var_forecasts, "Return": realised},
                       index=used_dates)
    out["Violation"] = out["Return"] < -out["VaR"]
    return out


def kupiec_test(violations, alpha=0.01):
    """Unconditional coverage (Kupiec 1995). H0: violation rate == alpha."""
    n = len(violations)
    x = int(violations.sum())
    pi_hat = x / n

    if x == 0:
        lr = -2 * n * np.log(1 - alpha)
    else:
        lr = -2 * ((n - x) * np.log(1 - alpha) + x * np.log(alpha)
                   - (n - x) * np.log(1 - pi_hat) - x * np.log(pi_hat))
    return {"N": n, "Violations": x, "Expected": n * alpha,
            "Rate": pi_hat, "LR_uc": lr,
            "p-value": 1 - stats.chi2.cdf(lr, 1)}


def christoffersen_test(violations):
    """Independence (Christoffersen 1998).

    H0: a violation today is independent of one yesterday. Clustering
    means the model updates too slowly when volatility rises -- failure
    concentrated in exactly the periods risk management cares about.
    """
    v = violations.astype(int).values
    n00 = int(((v[:-1] == 0) & (v[1:] == 0)).sum())
    n01 = int(((v[:-1] == 0) & (v[1:] == 1)).sum())
    n10 = int(((v[:-1] == 1) & (v[1:] == 0)).sum())
    n11 = int(((v[:-1] == 1) & (v[1:] == 1)).sum())

    pi01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)

    lg = lambda x: np.log(x) if x > 0 else 0.0
    ll_null = (n00 + n10) * lg(1 - pi) + (n01 + n11) * lg(pi)
    ll_alt = (n00 * lg(1 - pi01) + n01 * lg(pi01)
              + n10 * lg(1 - pi11) + n11 * lg(pi11))
    lr = -2 * (ll_null - ll_alt)

    return {"n01": n01, "n11": n11,
            "P(viol|no viol)": pi01, "P(viol|viol)": pi11,
            "LR_ind": lr, "p-value": 1 - stats.chi2.cdf(lr, 1)}
