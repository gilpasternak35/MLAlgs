from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class BaseClassifier(ABC):
    """Represents an abstract"""
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, features: Any, labels: Any) -> BaseClassifier:
        pass

    @abstractmethod
    def predict(self, data):
        pass
