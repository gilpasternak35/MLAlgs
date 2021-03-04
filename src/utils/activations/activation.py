from abc import ABC, abstractmethod
import numpy as np


class Activation(ABC):
    """Parent class for Activation functions"""

    @abstractmethod
    def compute(self, data: np.array):
        """Returns computed activation on a numpy array of data"""
        raise NotImplementedError
