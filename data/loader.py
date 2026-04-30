import pandas as pd
import numpy as np
import yfinance as yf

def get_prices(tickers, start, end=None, min_coverage=0.9):
    data = yf.download(tickers, start=start, end=end, progress=False)
    prices = data["Close"]
    thresh = int(len(prices) * min_coverage)
    prices = prices.dropna(axis=1, thresh=thresh)
    prices = prices.dropna(how="all")
    return prices.squeeze()   
