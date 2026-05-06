import numpy as np
from utils.metrics import to_returns, PerformanceStats
from utils.plot import plot_equity

class Backtest:
    def __init__(self, strategy, prices, transaction_cost=0.0):
        self.strategy = strategy
        self.prices = prices
        self.transaction_cost = transaction_cost
        self.results = None
    
    def run(self):
        positions = self.strategy.run(self.prices)
        returns = to_returns(self.prices)
        pnl = (positions.shift(1) * returns)
        if hasattr(pnl, 'columns'):
            pnl = pnl.sum(axis=1)
        
        costs = positions.diff().abs()
        if hasattr(costs, 'columns'):
            costs = costs.sum(axis=1)
        costs = costs * self.transaction_cost

        gross_exposure = positions.shift(1).abs()
        if hasattr(gross_exposure, 'columns'):
            gross_exposure = gross_exposure.sum(axis=1)
        
        ret = (pnl - costs) / gross_exposure
        ret = ret.replace([np.inf, -np.inf], np.nan).fillna(0)
        self.results = PerformanceStats(ret)
    
    def plot(self):
        plot_equity(self.results.ret, title=self.strategy.name)
