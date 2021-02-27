from typing import Any

from src.models.supervised.classifier.base_classifier import BaseClassifier


class KMeans(BaseClassifier):
    """An implementation of the unsupervised KMeans Clustering algorithm"""

    def __init__(self):
        """Instantiates a new KMeans classifier"""
        super().__init__()

    def fit(self, features: Any, labels: Any) -> BaseClassifier:
        """Fit the classifier to new data"""
        pass

    def predict(self, data):
        """Predict on new data"""
        pass