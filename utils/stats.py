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

def hurst(series):
    ...

def half_life(series):
    ...

def johansen_test(prices):
    ...