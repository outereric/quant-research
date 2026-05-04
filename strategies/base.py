from abc import ABC, abstractmethod

class Strategy(ABC):

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def generate_signals(self, prices):
        pass

    @abstractmethod
    def generate_positions(self, signals):
        pass

    def run(self, prices):
        signals = self.generate_signals(prices)
        positions = self.generate_positions(signals)
        return positions
