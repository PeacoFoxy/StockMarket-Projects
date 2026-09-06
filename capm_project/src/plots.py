import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


def plot_scatter_regression(excess_stock, excess_market, save_path=None):
    """Scatter plot of stock vs market excess returns with regression lines."""
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for i, ticker in enumerate(excess_stock.columns):
        ax = axes[i]
        x = excess_market
        y = excess_stock[ticker]

        ax.scatter(x, y, alpha=0.3, s=5, color='steelblue')

        m = sm.OLS(y, sm.add_constant(x)).fit()
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = m.params['const'] + m.params['SPX'] * x_line
        ax.plot(x_line, y_line, color='red', linewidth=2)

        ax.set_title(f"{ticker} (Beta = {m.params['SPX']:.2f})")
        ax.set_xlabel('Market Excess Return')
        ax.set_ylabel('Stock Excess Return')

    plt.suptitle('CAPM: Stock vs Market Excess Returns', fontsize=14, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_residuals(excess_stock, excess_market, returns_index, save_path=None):
    """Plot regression residuals over time for each stock."""
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for i, ticker in enumerate(excess_stock.columns):
        y = excess_stock[ticker]
        model = sm.OLS(y, sm.add_constant(excess_market)).fit()
        residuals = model.resid

        ax = axes[i]
        ax.scatter(returns_index, residuals, alpha=0.3, s=3, color='steelblue')
        ax.axhline(y=0, color='red', linewidth=1)
        ax.set_title(ticker)
        ax.set_ylabel('Residuals')

    plt.suptitle('CAPM Residuals Over Time', fontsize=14, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_rolling_beta(excess_stock, excess_market, window=60, save_path=None):
    """Plot rolling beta over time for each stock."""
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for i, ticker in enumerate(excess_stock.columns):
        rolling_beta = []
        dates = []

        for t in range(window, len(excess_market)):
            y = excess_stock[ticker].iloc[t-window:t]
            x = sm.add_constant(excess_market.iloc[t-window:t])
            model = sm.OLS(y, x).fit()
            rolling_beta.append(model.params['SPX'])
            dates.append(excess_market.index[t])

        ax = axes[i]
        ax.plot(dates, rolling_beta, color='steelblue', linewidth=0.8)
        ax.axhline(y=1, color='red', linestyle='--', linewidth=1)
        ax.set_title(ticker)
        ax.set_ylabel('Beta')

    plt.suptitle(f'Rolling {window}-Day Beta', fontsize=14, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_beta_bar(results_df, save_path=None):
    """Horizontal bar chart comparing betas across stocks."""
    betas = results_df['Beta'].sort_values(ascending=True)

    plt.figure(figsize=(10, 6))
    colors = ['steelblue' if b < 1 else 'coral' for b in betas]
    plt.barh(betas.index, betas.values, color=colors)
    plt.axvline(x=1, color='red', linestyle='--', linewidth=1, label='Beta = 1 (Market)')
    plt.xlabel('Beta')
    plt.title('CAPM Beta Comparison Across Stocks')
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_sml(results_df, excess_stock, excess_market, rf_daily, save_path=None):
    """Plot the Security Market Line with theoretical prediction."""
    avg_excess_returns = excess_stock.mean() * 252

    plt.figure(figsize=(10, 6))
    plt.scatter(results_df['Beta'], avg_excess_returns, color='steelblue', s=100, zorder=5)

    for ticker in results_df.index:
        plt.annotate(ticker, (results_df.loc[ticker, 'Beta'], avg_excess_returns[ticker]),
                     textcoords='offset points', xytext=(8, 5), fontsize=10)

    rf_annual = rf_daily.mean() * 252
    avg_market_excess = excess_market.mean() * 252
    x_line = np.linspace(0, 1.5, 100)
    y_line = rf_annual + x_line * avg_market_excess
    plt.plot(x_line, y_line, color='red', linestyle='--', linewidth=2, label='Theoretical SML')

    plt.xlabel('Beta')
    plt.ylabel('Annualized Excess Return')
    plt.title('Security Market Line (SML)')
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
