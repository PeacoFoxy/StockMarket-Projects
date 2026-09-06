import yfinance as yf
import pandas as pd
import pandas_datareader.data as web


def download_stock_prices(tickers, start_date, end_date):
    """Download adjusted close prices for a list of tickers."""
    stock_data = yf.download(tickers, start=start_date, end=end_date)['Close']
    return stock_data


def download_market_prices(market_ticker, start_date, end_date):
    """Download adjusted close prices for the market index."""
    market_data = yf.download(market_ticker, start=start_date, end=end_date)['Close']
    return market_data


def download_risk_free_rate(start_date, end_date):
    """Download 3-Month Treasury Bill rate from FRED."""
    rf_data = web.DataReader('DGS3MO', 'fred', start_date, end_date)
    return rf_data


def compute_returns(prices):
    """Compute daily simple returns from price DataFrame."""
    returns = prices.pct_change().dropna()
    return returns


def compute_daily_rf(rf_data, returns_index):
    """Convert annualized FRED rate to daily decimal and align with returns dates."""
    rf_daily = rf_data['DGS3MO'] / 100 / 252
    rf_daily = rf_daily.reindex(returns_index).ffill().dropna()
    return rf_daily


def compute_excess_returns(returns, rf_daily):
    """Subtract daily risk-free rate from all return columns."""
    excess_returns = returns.subtract(rf_daily, axis=0)
    excess_stock = excess_returns.drop(columns='SPX')
    excess_market = excess_returns['SPX']
    return excess_stock, excess_market


def sanity_check(stock_data, market_data):
    """Print basic data quality checks."""
    print(f"Stock data shape: {stock_data.shape}")
    print(f"Market data shape: {market_data.shape}")
    print(f"Date range: {stock_data.index[0]} to {stock_data.index[-1]}")
    print(f"\nMissing values per stock:")
    print(stock_data.isnull().sum())
