"""GARCH estimation and parameter extraction.

Wraps the `arch` package so that the notebook deals in results tables
rather than in model objects, and so that the same fitting logic is
reused across specifications in later stages.
"""

import numpy as np
import pandas as pd
from arch import arch_model


def fit_garch(series, p=1, q=1, dist="normal", mean="Constant", vol="GARCH", o=0):
    """Fit a single conditional volatility model by maximum likelihood.

    Parameters mirror the `arch` API so that GJR and EGARCH can reuse
    this function in Stage 5 without modification.

    A constant mean is retained throughout. Section 2.5 found statistically
    significant but economically negligible return autocorrelation
    (average |rho| about 0.04), attributable largely to non-synchronous
    trading in index constituents. An AR term would explain under 0.2% of
    return variance and leave the residuals essentially unchanged.

    `disp="off"` suppresses the optimiser's iteration log, which would
    otherwise print several hundred lines per fit.
    """
    model = arch_model(series, mean=mean, vol=vol, p=p, o=o, q=q, dist=dist)
    return model.fit(disp="off")


def fit_all(returns, **kwargs):
    """Fit the same specification to every index.

    Returns a dict keyed by index label so downstream functions can
    address results by name rather than by position.
    """
    results = {}
    for label, r in returns.items():
        results[label] = fit_garch(r, **kwargs)
        print(f"{label:8s} converged: {results[label].convergence_flag == 0}")
    return results


def half_life(persistence):
    """Days for a volatility shock to decay to half its initial size.

    Solves beta_total^h = 0.5 for h, where beta_total = alpha + beta.

    Persistence is hard to read directly: 0.985 and 0.995 look similar but
    imply half-lives of 46 and 138 days. The transformation makes the
    economic difference legible.
    """
    if persistence >= 1:
        return np.inf          # non-stationary: shocks never decay
    return np.log(0.5) / np.log(persistence)


def summarise_fits(results):
    """Collect estimated parameters into one comparable table.

    Columns are chosen to answer the project's central question directly:
    persistence and its half-life sit next to the components that
    generate them.
    """
    rows = []
    for label, res in results.items():
        pars = res.params

        omega = pars.get("omega", np.nan)
        alpha = pars.get("alpha[1]", np.nan)
        beta = pars.get("beta[1]", np.nan)
        persistence = alpha + beta

        # Unconditional (long-run) variance implied by the model:
        #   sigma^2 = omega / (1 - alpha - beta)
        # Defined only under stationarity. Annualised and square-rooted
        # for comparison with the realised volatilities in Section 1.3.
        if persistence < 1:
            uncond_var = omega / (1 - persistence)
            uncond_vol = np.sqrt(uncond_var * 252)
        else:
            uncond_vol = np.nan

        rows.append({
            "Index": label,
            "omega": omega,
            "alpha": alpha,
            "beta": beta,
            "alpha+beta": persistence,
            "Half-life (days)": half_life(persistence),
            "Uncond. vol (%)": uncond_vol,
            "Log-lik": res.loglikelihood,
            "AIC": res.aic,
            "BIC": res.bic,
        })

    return pd.DataFrame(rows).set_index("Index").round(4)


def parameter_significance(results):
    """Standard errors and t-statistics for the estimated parameters.

    Point estimates alone are not enough: a persistence of 0.99 estimated
    with a standard error of 0.05 supports a very different conclusion
    from the same figure estimated with a standard error of 0.003.
    """
    rows = []
    for label, res in results.items():
        for name in res.params.index:
            rows.append({
                "Index": label,
                "Parameter": name,
                "Estimate": res.params[name],
                "Std. Error": res.std_err[name],
                "t-stat": res.tvalues[name],
                "p-value": res.pvalues[name],
            })
    return pd.DataFrame(rows).set_index(["Index", "Parameter"]).round(4)
