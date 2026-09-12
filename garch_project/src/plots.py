"""Exploratory plots for the GARCH project.

Every function takes the dict-of-Series structure produced by
`data_loader.compute_log_returns` and accepts an optional `save_path`
so that figures can be written to outputs/ from the notebook.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Crisis windows used to annotate the time-series plots. Shading these
# turns "volatility spikes somewhere around 2008" into a claim the reader
# can verify against specific events.
CRISES = {
    "GFC": ("2008-09-01", "2009-06-30"),
    "EU debt": ("2011-07-01", "2012-07-31"),
    "COVID": ("2020-02-20", "2020-04-30"),
    "Inflation": ("2022-01-01", "2022-10-31"),
}


def _shade_crises(ax):
    """Shade crisis windows on a time-axis plot."""
    for name, (start, end) in CRISES.items():
        ax.axvspan(np.datetime64(start), np.datetime64(end),
                   color="grey", alpha=0.15, zorder=0)


def plot_prices(prices, save_path=None):
    """Normalised price levels -- a first check on data integrity.

    Prices are rebased to 100 at the sample start so that four indices on
    very different scales can share one axis. Discontinuities, flat
    stretches or implausible jumps would show up here before they
    contaminate anything downstream.
    """
    fig, ax = plt.subplots(figsize=(13, 5))
    for label, s in prices.items():
        ax.plot(s.index, 100 * s / s.iloc[0], linewidth=1.1, label=label)
    _shade_crises(ax)
    ax.set_title("Index levels, rebased to 100 at 2007-04-02")
    ax.set_ylabel("Index (base = 100)")
    ax.legend(loc="upper left")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_returns(returns, save_path=None):
    """Return series -- the visual signature of volatility clustering.

    A constant-variance series would look like a band of uniform
    thickness. Equity returns instead alternate between calm stretches
    and violent ones, which is precisely what GARCH is built to model.
    """
    n = len(returns)
    fig, axes = plt.subplots(n, 1, figsize=(13, 2.6 * n), sharex=True)

    for ax, (label, r) in zip(axes, returns.items()):
        ax.plot(r.index, r.values, linewidth=0.4, color="steelblue")
        _shade_crises(ax)
        ax.set_ylabel(f"{label} (%)")
        ax.axhline(0, color="black", linewidth=0.5)

    axes[0].set_title("Daily log returns (%) -- shaded regions mark crisis periods")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_rolling_volatility(returns, window=21, save_path=None):
    """Annualised rolling volatility.

    Converts the clustering visible in the return plot into a level that
    can be read off directly. A 21-day window is one trading month --
    short enough to react to regime shifts, long enough to be stable.
    """
    fig, ax = plt.subplots(figsize=(13, 5))
    for label, r in returns.items():
        # sqrt-of-time rule; r is already in percent, so no further scaling
        vol = r.rolling(window).std() * np.sqrt(252)
        ax.plot(vol.index, vol.values, linewidth=0.9, label=label)
    _shade_crises(ax)
    ax.set_title(f"{window}-day rolling volatility, annualised (%)")
    ax.set_ylabel("Volatility (%)")
    ax.legend(loc="upper left")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_return_distributions(returns, save_path=None):
    """Histogram against a fitted normal, plus a normal QQ plot.

    The histogram shows the peak and shoulders; the QQ plot is the
    sharper diagnostic for the tails. Under normality the points would
    lie on the 45-degree line. Departures at both ends -- the
    characteristic S-shape -- indicate excess kurtosis.
    """
    n = len(returns)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 7))

    for j, (label, r) in enumerate(returns.items()):
        # Top row: empirical density vs fitted normal
        ax = axes[0, j]
        ax.hist(r, bins=150, density=True, alpha=0.6,
                color="steelblue", edgecolor="none")
        x = np.linspace(r.min(), r.max(), 500)
        ax.plot(x, stats.norm.pdf(x, r.mean(), r.std()),
                color="red", linewidth=1.2, label="Normal")
        ax.set_xlim(-6, 6)          # zoom on the body; tails are the QQ plot's job
        ax.set_title(label)
        if j == 0:
            ax.set_ylabel("Density")
        ax.legend(fontsize=8)

        # Bottom row: normal QQ plot
        ax = axes[1, j]
        stats.probplot(r, dist="norm", plot=ax)
        ax.set_title("")
        ax.get_lines()[0].set_markersize(2)
        ax.get_lines()[0].set_color("steelblue")
        ax.get_lines()[1].set_color("red")
        if j == 0:
            ax.set_ylabel("Sample quantiles")

    fig.suptitle("Return distributions vs the normal benchmark", y=1.00)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_rolling_volatility_panels(returns, window=21, save_path=None):
    """Rolling volatility, one panel per index on a shared scale.

    The overlaid version is useful for ranking levels at a glance but
    becomes unreadable during calm periods when the four series overlap.
    Separate panels sacrifice direct comparison for legibility; a shared
    y-axis keeps the magnitudes comparable across panels.
    """
    n = len(returns)
    fig, axes = plt.subplots(n, 1, figsize=(13, 2.6 * n), sharex=True, sharey=True)

    for ax, (label, r) in zip(axes, returns.items()):
        vol = r.rolling(window).std() * np.sqrt(252)
        ax.plot(vol.index, vol.values, linewidth=0.8, color="steelblue")

        # Unconditional annualised volatility as a reference line: it makes
        # visible how rarely realised volatility actually sits at its average.
        uncond = r.std() * np.sqrt(252)
        ax.axhline(uncond, color="red", linestyle="--", linewidth=0.9,
                   label=f"Unconditional: {uncond:.1f}%")

        _shade_crises(ax)
        ax.set_ylabel(f"{label} (%)")
        ax.legend(loc="upper right", fontsize=8)

    axes[0].set_title(f"{window}-day rolling volatility, annualised (%)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_conditional_volatility(results, returns=None, window=21, save_path=None):
    """Fitted conditional volatility, optionally against a rolling estimate.

    The comparison is the point. Both series describe the same underlying
    quantity, but the rolling standard deviation weights the last `window`
    days equally and ignores everything before them, while GARCH weights
    all history with geometrically declining weights. Two consequences are
    visible: GARCH responds on the day a shock arrives rather than easing
    into it, and it decays smoothly instead of dropping discontinuously
    when a large observation leaves the window.
    """
    n = len(results)
    fig, axes = plt.subplots(n, 1, figsize=(13, 2.8 * n), sharex=True)

    for ax, (label, res) in zip(axes, results.items()):
        # res.conditional_volatility is the fitted sigma_t series, in the
        # same percentage units as the input returns.
        cond_vol = res.conditional_volatility * np.sqrt(252)

        if returns is not None:
            roll = returns[label].rolling(window).std() * np.sqrt(252)
            ax.plot(roll.index, roll.values, linewidth=0.7,
                    color="grey", alpha=0.7, label=f"{window}-day rolling")

        ax.plot(cond_vol.index, cond_vol.values, linewidth=0.8,
                color="steelblue", label="GARCH(1,1)")

        _shade_crises(ax)
        ax.set_ylabel(f"{label} (%)")
        ax.legend(loc="upper right", fontsize=8)

    axes[0].set_title("Conditional volatility, annualised (%)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_residual_qq(results_normal, results_t, save_path=None):
    """QQ plots of standardised residuals against their assumed distributions.

    Each model is checked against the distribution it actually assumes:
    normal residuals against a normal, t residuals against a t with the
    estimated degrees of freedom. Points on the line mean the assumption
    holds.
    """
    n = len(results_normal)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 7))

    for j, label in enumerate(results_normal.keys()):
        # Top row: normal model vs normal quantiles
        z_n = results_normal[label].std_resid.dropna()
        ax = axes[0, j]
        stats.probplot(z_n, dist="norm", plot=ax)
        ax.set_title(f"{label} — Normal")
        ax.get_lines()[0].set_markersize(2)
        ax.get_lines()[0].set_color("steelblue")
        ax.get_lines()[1].set_color("red")
        ax.set_xlabel("")
        if j > 0:
            ax.set_ylabel("")

        # Bottom row: t model vs t quantiles with the ESTIMATED nu.
        # Using a normal reference here would be the wrong benchmark --
        # the t model never claimed its residuals were Gaussian.
        z_t = results_t[label].std_resid.dropna()
        nu = results_t[label].params["nu"]
        ax = axes[1, j]
        stats.probplot(z_t, dist=stats.t, sparams=(nu,), plot=ax)
        ax.set_title(f"{label} — t({nu:.1f})")
        ax.get_lines()[0].set_markersize(2)
        ax.get_lines()[0].set_color("steelblue")
        ax.get_lines()[1].set_color("red")
        if j > 0:
            ax.set_ylabel("")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_news_impact(results, save_path=None):
    """News impact curve: conditional variance as a function of yesterday's shock.

    Holds sigma^2_{t-1} at its unconditional level and traces sigma^2_t
    across a range of epsilon_{t-1}. Under symmetric GARCH the curve is a
    parabola centred at zero; GJR bends it, producing a steeper left arm.

    Engle and Ng (1993) introduced this as the standard way to compare
    asymmetric specifications visually.
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))
    eps = np.linspace(-5, 5, 400)

    for label, res in results.items():
        p = res.params
        a, g, b = p["alpha[1]"], p.get("gamma[1]", 0.0), p["beta[1]"]
        omega = p["omega"]

        persist = a + g / 2 + b
        sigma2_bar = omega / (1 - persist)

        sigma2 = omega + a * eps**2 + g * eps**2 * (eps < 0) + b * sigma2_bar
        ax.plot(eps, sigma2, linewidth=1.6, label=label)

    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel(r"Shock yesterday, $\epsilon_{t-1}$ (%)")
    ax.set_ylabel(r"Conditional variance today, $\sigma_t^2$")
    ax.set_title("News impact curves, GJR-skewt")
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
