from typing import Any

import numpy as np

from src.models.nn.perceptron.perceptron import Perceptron
from src.models.supervised.classifier.base_classifier import BaseClassifier


class AveragedPerceptron(Perceptron):
    """Implemented Model for an Averaged Perceptron"""
    def __init__(self, max_iter: int):
        super.__init__(max_iter)

    def fit(self, features: Any, labels: Any) -> BaseClassifier:
        pass


