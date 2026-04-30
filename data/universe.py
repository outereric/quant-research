import pandas as pd
import numpy as np
import re

def get_sp500_current():
    url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
    tickers = pd.read_csv(url)["Symbol"].tolist()
    return [t.replace(".", "-") for t in tickers]

def get_sp500_historical(date):
    url = "https://raw.githubusercontent.com/fja05680/sp500/master/S%26P%20500%20Historical%20Components%20%26%20Changes.csv"
    hist = pd.read_csv(url)
    hist["date"] = pd.to_datetime(hist["date"])
    target = pd.Timestamp(date)
    row = hist[hist["date"] <= target].iloc[-1]
    tickers = row["tickers"].split(",")
    tickers = [t for t in tickers if not re.search(r"-\d{6}$", t)]
    tickers = [t.replace(".", "-") for t in tickers]
    return tickers
