import pandas as pd
import numpy as np

TRADING_DAYS_PER_YEAR = 252
     
def to_returns(prices):
    return prices / prices.shift(1) - 1

def to_log_returns(prices):
    return np.log1p(prices.pct_change())

def annualized_return(returns, periods=TRADING_DAYS_PER_YEAR):
    return returns.mean() * periods

def annualized_vol(returns, periods=TRADING_DAYS_PER_YEAR):
    return returns.std() * np.sqrt(periods)

def sharpe(returns, rf=0.0, periods=TRADING_DAYS_PER_YEAR):
    er = returns.mean() - rf / periods
    std = returns.std()
    if std == 0:
        return np.nan
    return np.sqrt(periods) * er / std

def equity(returns):
    return (1 + returns).cumprod()

def max_drawdown(returns):
    _equity = equity(returns)
    return (_equity / _equity.cummax() - 1).min()

class PerformanceStats:
    def __init__(self, ret):
        self.ret = ret
        self.ann_ret = annualized_return(ret)
        self.ann_vol = annualized_vol(ret)
        self.sharpe = sharpe(ret)
        self.equity = equity(ret)
        self.max_drawdown = max_drawdown(ret)
    
    def display(self):
        print(f"Ann Return:  {self.ann_ret:.2%}")
        print(f"Ann Vol:     {self.ann_vol:.2%}")
        print(f"Sharpe:      {self.sharpe:.4f}")
        print(f"Max DD:      {self.max_drawdown:.2%}")

