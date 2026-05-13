import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen

def adf_test(series):
    result = adfuller(series, regression="c", autolag="AIC")
    return ({
        "statistic": result[0],
        "p_value": result[1],
        "critical_values": result[4], # critical values [90%, 95%, 99%]
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
    result = coint_johansen(prices, det_order=0, k_ar_diff=1)
    return ({
        "trace_stat": result.lr1[0],        # test statistic for rank=0
        "critical_values": result.cvt[0], # critical values [90%, 95%, 99%]
        "cointegrating_vector": result.evec[:, 0], # hedge ratio
    })