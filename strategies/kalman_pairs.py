import pandas as pd
import numpy as np
from strategies.base import Strategy

def kalman_filter(x, y, delta=1e-5, R=1e-3):
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(y)

    # Process noise
    Q = delta / (1 - delta) * np.eye(2)

    # Output arrays
    theta = np.full((2, n), np.nan) # rows: [beta, alpha]
    e = np.full(n, np.nan) # measurement prediction error
    S = np.full(n, np.nan) # measurement variance prediction

    # Initialise state
    theta[:, 0] = 0.0
    P = np.eye(2)

    for t in range(n):
        if t > 0:
            theta[:, t] = theta[:, t - 1] # state prediction
            P = P + Q # state covariance prediction

        F = np.array([x[t], 1.0]) # observation matrix

        e[t] = y[t] - F @ theta[:, t] # measurement prediction error

        S[t] = F @ P @ F + R # measurement variance prediction

        K = P @ F / S[t] # Kalman gain

        theta[:, t] = theta[:, t] + K * e[t] # state update
       
        I = np.eye(2)
        P = (I - np.outer(K, F)) @ P # state covariance update

    return pd.DataFrame({
        "beta"   : theta[0],
        "alpha"  : theta[1],
        "spread" : e,
        "sqrt_S" : np.sqrt(S),
        "zscore" : e / np.sqrt(S),
    })

class KalmanPairs(Strategy):
    def __init__(self, entry_zscore=1.0, exit_zscore=0.0):
        super().__init__(name="KalmanPairs")
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore

    def generate_signals(self, prices):
        self.col_x = prices.columns[0]
        self.col_y = prices.columns[1]
        signals = kalman_filter(prices[self.col_x], prices[self.col_y])
        signals.index = prices.index
        return signals

    def generate_positions(self, signals):
        zscore = signals["zscore"]
        longs_entry = zscore < -self.entry_zscore
        longs_exit = zscore >= self.exit_zscore
        shorts_entry = zscore > self.entry_zscore
        shorts_exit = zscore <= self.exit_zscore

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

        num_units = num_units_long + num_units_shorts

        positions = pd.DataFrame(index=signals.index)
        positions[self.col_x] = -num_units * signals["beta"]
        positions[self.col_y] = num_units

        return positions

