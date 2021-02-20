from abc import ABC, abstractmethod


class Cost(ABC):
    """Represents a basic cost function for an ML model"""

    def __init__(self):
        pass

    @staticmethod
    @abstractmethod
    def compute(self, *args, **kwargs) -> float:
        pass
