"""Data acquisition and return computation for the GARCH project."""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

# Yahoo Finance ticker -> 项目内统一短标签
INDICES = {
    "^GSPC": "SPX",        # S&P 500 (US)
    "^STOXX50E": "SX5E",   # EURO STOXX 50 (Eurozone)
    "^N225": "Nikkei",     # Nikkei 225 (Japan)
    "^FTSE": "FTSE",       # FTSE 100 (UK)
}


def download_index_prices(tickers=None, start="2005-01-01", end=None):
    """Download daily prices, one independent series per index."""
    tickers = tickers or INDICES
    prices = {}                        # 用 dict 装 Series,不用 DataFrame

    for ticker, label in tickers.items():
        df = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=True,          # 显式写出,不依赖版本默认值
            progress=False,            # 关进度条,输出更干净
        )
        # .squeeze() 把单列 DataFrame 压成 Series(防 MultiIndex 列)
        s = df["Close"].squeeze().dropna()
        s.name = label
        prices[label] = s

        # 每一步都要有可验证的输出 -- 数据有缺口能立刻发现
        print(f"{label:8s} {len(s):>5,} obs   {s.index[0].date()} -> {s.index[-1].date()}")

    return prices


def compute_log_returns(prices, scale=100.0):
    """Convert prices to percentage log returns."""
    returns = {}
    for label, s in prices.items():
        # np.log(s).diff() == log(P_t / P_{t-1})
        # scale=100 是 MLE 优化器的数值稳定性要求,不是金融理由
        r = scale * np.log(s).diff().dropna()   # 首个值必为 NaN
        r.name = label
        returns[label] = r
    return returns


def summary_statistics(returns):
    """Descriptive statistics -- the empirical case for GARCH."""
    rows = []
    for label, r in returns.items():
        jb_stat, jb_pval = stats.jarque_bera(r)      # H0: 正态分布
        rows.append({
            "Index": label,
            "N": len(r),
            "Start": r.index[0].date(),
            "End": r.index[-1].date(),
            "Mean": r.mean(),
            "Std": r.std(),
            "Ann. Vol (%)": r.std() * np.sqrt(252),  # sqrt-of-time rule
            "Skew": stats.skew(r),                   # 股指通常为负
            "Excess Kurt": stats.kurtosis(r),        # Fisher: 正态 = 0
            "Min": r.min(),
            "Max": r.max(),
            "JB p-value": jb_pval,
        })
    return pd.DataFrame(rows).set_index("Index").round(4)


def save_returns(returns, path="../data/index_returns.csv"):
    """Persist returns as a wide CSV. NaNs are expected (different calendars)."""
    df = pd.DataFrame(returns)         # 不同日历自动 outer join,缺口填 NaN
    df.to_csv(path)                    # 读回来按列单独 dropna,别整表 dropna()
    return df
