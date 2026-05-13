import pandas as pd
import numpy as np
from strategies.base import Strategy

class Bollinger(Strategy):
    def __init__(self, lookback=20, entry_zscore=1.0, exit_zscore=0.0):
        super().__init__(name="BollingerBand")
        self.lookback = lookback
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
    
    def generate_signals(self, prices):
        rolling_mean = prices.rolling(self.lookback).mean()
        rolling_std = prices.rolling(self.lookback).std()
        zscore = (prices - rolling_mean) / rolling_std
        return zscore
    
    def generate_positions(self, signals):
        longs_entry = signals < -self.entry_zscore
        longs_exit = signals >= self.exit_zscore
        shorts_entry = signals > self.entry_zscore
        shorts_exit = signals <= self.exit_zscore

        num_units_long = pd.Series(np.nan, index=signals.index)
        num_units_long.iloc[0] = 0
        num_units_long[longs_entry] = 1
        num_units_long[longs_exit] = 0
        num_units_long = num_units_long.ffill().fillna(0)

        num_units_shorts = pd.Series(np.nan, index=signals.index)
        num_units_shorts.iloc[0] = 0
        num_units_shorts[shorts_entry] = -1
        num_units_shorts[shorts_exit] = 0
        num_units_shorts = num_units_shorts.ffill().fillna(0)

        return num_units_long + num_units_shorts