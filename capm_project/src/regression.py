import pandas as pd
import numpy as np
import statsmodels.api as sm


def run_capm_regression(excess_stock, excess_market):
    """Run CAPM regression for all stocks. Returns summary DataFrame."""
    X = sm.add_constant(excess_market)
    results = {}

    for ticker in excess_stock.columns:
        y = excess_stock[ticker]
        model = sm.OLS(y, X).fit()
        results[ticker] = {
            'Alpha': model.params['const'],
            'Beta': model.params['SPX'],
            'R-squared': model.rsquared,
            'Alpha_pval': model.pvalues['const'],
            'Beta_pval': model.pvalues['SPX']
        }

    results_df = pd.DataFrame(results).T
    results_df = results_df.round(4)
    return results_df


def run_detailed_summary(excess_stock, excess_market, ticker='AAPL'):
    """Print full OLS regression summary for a single stock."""
    y = excess_stock[ticker]
    X = sm.add_constant(excess_market)
    model = sm.OLS(y, X).fit()
    print(model.summary())
    return model


def run_full_summary(excess_stock, excess_market):
    """Run regression with extended statistics for all stocks."""
    X = sm.add_constant(excess_market)
    summary_list = []

    for ticker in excess_stock.columns:
        y = excess_stock[ticker]
        model = sm.OLS(y, X).fit()
        summary_list.append({
            'Stock': ticker,
            'Alpha': model.params['const'],
            'Alpha_tstat': model.tvalues['const'],
            'Alpha_pval': model.pvalues['const'],
            'Beta': model.params['SPX'],
            'Beta_tstat': model.tvalues['SPX'],
            'Beta_pval': model.pvalues['SPX'],
            'R-squared': model.rsquared,
            'F_stat': model.fvalue,
            'Durbin_Watson': sm.stats.stattools.durbin_watson(model.resid),
            'N_obs': int(model.nobs)
        })

    summary_df = pd.DataFrame(summary_list).set_index('Stock')
    summary_df = summary_df.round(4)
    return summary_df


def compare_time_windows(returns, excess_stock, excess_market):
    """Compare CAPM betas across pre-COVID, post-COVID, and full period."""
    periods = {
        'Full (2015-2025)': ('2015-01-01', '2025-12-31'),
        'Pre-COVID (2015-2019)': ('2015-01-01', '2019-12-31'),
        'Post-COVID (2020-2025)': ('2020-01-01', '2025-12-31'),
    }

    comparison = []

    for period_name, (start, end) in periods.items():
        mask = (returns.index >= start) & (returns.index <= end)
        sub_excess_stock = excess_stock.loc[mask]
        sub_excess_market = excess_market.loc[mask]
        X = sm.add_constant(sub_excess_market)

        for ticker in sub_excess_stock.columns:
            y = sub_excess_stock[ticker]
            model = sm.OLS(y, X).fit()
            comparison.append({
                'Period': period_name,
                'Ticker': ticker,
                'Alpha': model.params['const'],
                'Beta': model.params['SPX'],
                'N-obs': int(model.nobs),
            })

    comparison_df = pd.DataFrame(comparison)
    comparison_pivot = comparison_df.pivot(index='Ticker', columns='Period', values='Beta')
    comparison_pivot = comparison_pivot.round(4)
    return comparison_pivot


def compare_daily_vs_monthly(prices, rf_daily, results_df):
    """Compare CAPM betas using daily vs monthly data."""
    monthly_prices = prices.resample('ME').last()
    monthly_returns = monthly_prices.pct_change().dropna()

    rf_monthly = rf_daily.resample('ME').last() * 21
    rf_monthly = rf_monthly.reindex(monthly_returns.index).ffill().dropna()
    monthly_returns = monthly_returns.loc[rf_monthly.index]

    monthly_excess = monthly_returns.subtract(rf_monthly, axis=0)
    monthly_excess_stock = monthly_excess.drop(columns='SPX')
    monthly_excess_market = monthly_excess['SPX']

    X_m = sm.add_constant(monthly_excess_market)
    monthly_results = {}

    for ticker in monthly_excess_stock.columns:
        y = monthly_excess_stock[ticker]
        model = sm.OLS(y, X_m).fit()
        monthly_results[ticker] = {
            'Beta_Monthly': model.params['SPX'],
            'Alpha_pval_Monthly': model.pvalues['const'],
            'R_sq_Monthly': model.rsquared
        }

    monthly_df = pd.DataFrame(monthly_results).T

    compare = pd.DataFrame({
        'Beta_Daily': results_df['Beta'],
        'Beta_Monthly': monthly_df['Beta_Monthly'],
        'Diff': monthly_df['Beta_Monthly'] - results_df['Beta'],
        'R_sq_Daily': results_df['R-squared'],
        'R_sq_Monthly': monthly_df['R_sq_Monthly']
    }).round(4)

    return compare
