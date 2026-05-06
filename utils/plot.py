import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.metrics import equity

def plot_equity(returns, title="Equity Curve"):
    _equity = equity(returns)
    _equity.plot(label=title)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_comparison(returns_dict, title="Strategy Comparison"):
    for name, ret in returns_dict.items():
        _equity = equity(ret)
        _equity.plot(label=name)
    
    plt.title(title)
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.grid(True)
    plt.tight_layout()
    plt.show()