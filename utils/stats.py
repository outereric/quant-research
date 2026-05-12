import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

def adf_test(series):
    result = adfuller(series, regression="c", autolag="AIC")
    return ({
        "statistic": result[0],
        "p_value": result[1],
        "critical_values": result[4], 
    })

def hurst(series, max_lag=100):
    series = np.asarray(series, float)
    series = series[np.isfinite(series)]

    lags = np.arange(2, max_lag)
    tau = []

    for lag in lags:
        diff = series[lag:] - series[:-lag]
        tau.append(np.var(diff))
    
    tau = np.asarray(tau)
    slope, intercept = np.polyfit(np.log(lags), np.log(tau), 1)
    return slope / 2

def half_life(series):
    series = np.asarray(series)
    series = series[np.isfinite(series)]
    lag = series[:-1]
    delta = series[1:] - series[:-1]

    slope, intercept = np.polyfit(lag, delta, 1)

    if slope >= 0:
        return np.nan

    return -np.log(2) / np.log(1 + slope)

def johansen_test(prices):
    ...