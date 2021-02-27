from abc import abstractmethod
from typing import Any

from src.models.supervised.classifier.base_classifier import BaseClassifier


class UnsupervisedBaseClassifier(BaseClassifier):
    """Unsupervised Classifier Base Class"""

    def __init__(self):
        """Instantiates a new KMeans classifier"""
        super().__init__()

    @abstractmethod
    def fit(self, features: Any) -> BaseClassifier:
        """Fit the classifier to new data"""
        pass

    @abstractmethod
    def predict(self, data):
        """Predict on new data"""
        pass