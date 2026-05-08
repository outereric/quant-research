import pandas as pd
import numpy as np

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Basic transforms
# ---------------------------------------------------------------------------

def to_returns(prices):
    return prices / prices.shift(1) - 1

def to_log_returns(prices):
    return np.log1p(prices.pct_change())


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------

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

def sortino(returns, rf=0.0, periods=TRADING_DAYS_PER_YEAR):
    """Sharpe but only penalises downside deviation."""
    er = returns.mean() - rf / periods
    downside = returns[returns < 0].std()
    if downside == 0:
        return np.nan
    return np.sqrt(periods) * er / downside

def equity(returns):
    return (1 + returns).cumprod()

def max_drawdown(returns):
    _equity = equity(returns)
    return (_equity / _equity.cummax() - 1).min()

def max_drawdown_duration(returns):
    """Number of calendar days in the longest drawdown period."""
    _equity = equity(returns)
    roll_max = _equity.cummax()
    underwater = _equity < roll_max

    max_dur = 0
    current_dur = 0
    for uw in underwater:
        if uw:
            current_dur += 1
            max_dur = max(max_dur, current_dur)
        else:
            current_dur = 0
    return max_dur

def calmar(returns, periods=TRADING_DAYS_PER_YEAR):
    """Annualised return divided by absolute max drawdown."""
    mdd = max_drawdown(returns)
    if mdd == 0:
        return np.nan
    return annualized_return(returns, periods) / abs(mdd)

def win_rate(returns):
    """Fraction of periods with a positive return."""
    valid = returns.dropna()
    if len(valid) == 0:
        return np.nan
    return (valid > 0).mean()

def avg_win(returns):
    wins = returns[returns > 0]
    return wins.mean() if len(wins) > 0 else np.nan

def avg_loss(returns):
    losses = returns[returns < 0]
    return losses.mean() if len(losses) > 0 else np.nan

def profit_factor(returns):
    """Gross profit / gross loss (absolute)."""
    gross_win = returns[returns > 0].sum()
    gross_loss = returns[returns < 0].sum()
    if gross_loss == 0:
        return np.nan
    return gross_win / abs(gross_loss)

def skewness(returns):
    return returns.skew()

def kurtosis(returns):
    """Excess kurtosis (normal distribution = 0)."""
    return returns.kurtosis()

def value_at_risk(returns, confidence=0.95):
    """Historical VaR — worst loss at the given confidence level."""
    return returns.quantile(1 - confidence)

def conditional_var(returns, confidence=0.95):
    """Expected Shortfall (CVaR) — average return below the VaR threshold."""
    threshold = value_at_risk(returns, confidence)
    tail = returns[returns <= threshold]
    return tail.mean() if len(tail) > 0 else np.nan


# ---------------------------------------------------------------------------
# PerformanceStats — container for all metrics
# ---------------------------------------------------------------------------

class PerformanceStats:
    def __init__(self, ret, turnover=None):
        self.ret = ret
        self.turnover = turnover  # optional daily turnover Series

        # Core
        self.ann_ret      = annualized_return(ret)
        self.ann_vol      = annualized_vol(ret)
        self.sharpe       = sharpe(ret)
        self.sortino      = sortino(ret)
        self.calmar       = calmar(ret)
        self.equity       = equity(ret)

        # Drawdown
        self.max_drawdown = max_drawdown(ret)
        self.max_dd_days  = max_drawdown_duration(ret)

        # Win/loss
        self.win_rate     = win_rate(ret)
        self.avg_win      = avg_win(ret)
        self.avg_loss     = avg_loss(ret)
        self.profit_factor = profit_factor(ret)

        # Tail risk
        self.skew         = skewness(ret)
        self.kurt         = kurtosis(ret)
        self.var_95       = value_at_risk(ret, 0.95)
        self.cvar_95      = conditional_var(ret, 0.95)

        # Turnover (annualised, if provided)
        if turnover is not None:
            self.avg_daily_turnover = turnover.mean()
            self.ann_turnover       = turnover.mean() * TRADING_DAYS_PER_YEAR
        else:
            self.avg_daily_turnover = None
            self.ann_turnover       = None

    def display(self):
        w = 28  # column width

        def row(label, value, fmt=".4f"):
            if value is None or (isinstance(value, float) and np.isnan(value)):
                print(f"  {label:<{w}} {'N/A':>10}")
            else:
                print(f"  {label:<{w}} {value:>10{fmt}}")

        def pct(label, value):
            row(label, value * 100 if value is not None and not (isinstance(value, float) and np.isnan(value)) else value, ".2f")

        sep = "  " + "-" * (w + 12)

        print()
        print("  ── Returns ─────────────────────────────")
        pct("Ann Return (%)",     self.ann_ret)
        pct("Ann Vol (%)",        self.ann_vol)

        print(sep)
        print("  ── Risk-Adjusted ───────────────────────")
        row("Sharpe",             self.sharpe)
        row("Sortino",            self.sortino)
        row("Calmar",             self.calmar)

        print(sep)
        print("  ── Drawdown ────────────────────────────")
        pct("Max Drawdown (%)",   self.max_drawdown)
        row("Max DD Duration (days)", self.max_dd_days, "d")

        print(sep)
        print("  ── Win / Loss ──────────────────────────")
        pct("Win Rate (%)",       self.win_rate)
        pct("Avg Win (%)",        self.avg_win)
        pct("Avg Loss (%)",       self.avg_loss)
        row("Profit Factor",      self.profit_factor)

        print(sep)
        print("  ── Tail Risk ───────────────────────────")
        row("Skewness",           self.skew)
        row("Excess Kurtosis",    self.kurt)
        pct("VaR 95% (%)",        self.var_95)
        pct("CVaR 95% (%)",       self.cvar_95)

        if self.ann_turnover is not None:
            print(sep)
            print("  ── Turnover ────────────────────────────")
            pct("Avg Daily Turnover (%)", self.avg_daily_turnover)
            pct("Ann Turnover (%)",       self.ann_turnover)

        print()
