import numpy as np
import pandas as pd
RF_ANNUAL = 0.045
TRADING_DAYS = 252


def sharpe_scores(returns_df, rf_annual=RF_ANNUAL, trading_days=TRADING_DAYS):
    rf_daily = rf_annual / trading_days
    mean_ret = returns_df.mean()
    std_ret = returns_df.std().replace(0, np.nan)
    return ((mean_ret - rf_daily) / std_ret).dropna().sort_values(ascending=False)


def select_top_by_sharpe(returns_df, top_n=10):
    scores = sharpe_scores(returns_df)
    return scores.head(top_n).index.tolist(), scores


def build_allocation_equal(symbols):
    return pd.DataFrame({"Asset": symbols, "Weight": np.ones(len(symbols)) / len(symbols)})


def build_allocation_80_20(train_returns):
    scores = sharpe_scores(train_returns)
    ranked = scores.reset_index(); ranked.columns = ["Asset", "Score"]
    n_assets = len(ranked)
    top_count = max(1, int(np.ceil(0.2 * n_assets)))
    bottom_count = n_assets - top_count
    top_weights = [0.8 / top_count] * top_count
    bottom_weights = [0.2 / bottom_count] * bottom_count if bottom_count > 0 else []
    ranked["Weight"] = top_weights + bottom_weights
    return ranked[["Asset", "Weight"]]


def port_char(weights_df, returns_df, annualize=True, freq=TRADING_DAYS):
    er = returns_df.mean().reset_index(); er.columns = ["Asset", "Er"]
    wm = pd.merge(weights_df, er, on="Asset", how="left").fillna({"Er": 0.0})
    portfolio_er_daily = np.dot(wm["Weight"], wm["Er"])
    cov_matrix = returns_df.cov().loc[wm["Asset"].tolist(), wm["Asset"].tolist()]
    w = wm["Weight"].values
    portfolio_std_daily = np.sqrt(np.dot(w, np.dot(cov_matrix, w)))
    if annualize:
        return portfolio_er_daily * freq, portfolio_std_daily * np.sqrt(freq)
    return portfolio_er_daily, portfolio_std_daily


def port_char_from_series(portfolio_return_series, annualize=True, freq=TRADING_DAYS):
    r = pd.Series(portfolio_return_series).dropna()
    er_daily, std_daily = r.mean(), r.std()
    if annualize:
        return er_daily * freq, std_daily * np.sqrt(freq)
    return er_daily, std_daily


def sharpe_port(weights_df, returns_df, rf=RF_ANNUAL):
    er, std = port_char(weights_df, returns_df, annualize=True)
    return (er - rf) / (std + 1e-12)


def sharpe_from_series(portfolio_return_series, rf=RF_ANNUAL):
    er, std = port_char_from_series(portfolio_return_series, annualize=True)
    return (er - rf) / (std + 1e-12)


def cumulative_from_weights(weights_df, returns_df):
    merged = weights_df.set_index("Asset")["Weight"].reindex(returns_df.columns).fillna(0)
    r = (returns_df * merged).sum(axis=1)
    return (1 + r).cumprod()
