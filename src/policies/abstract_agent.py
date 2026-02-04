from abc import ABC, abstractmethod
import numpy as np

class Agent(ABC):
    @abstractmethod
    def __call__(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        pass

    @abstractmethod
    def add(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool, cost: float) -> None:
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def save_checkpoint(self, path: str):
        pass

    @abstractmethod
    def load_checkpoint(self, path: str):
        pass
