from typing import Union

import numpy as np

from src.models.supervised.classifier.decision_tree.node import Node
from src.utils.cost import Cost


class DecisionTree(object):
    """
    Represents a randomly sampled decision tree algorithm
    """
    _MIN_DEPTH = 3

    def __init__(self,
                 data: np.ndarray,
                 labels: np.ndarray,
                 depth: Union[str, int] = 'random',
                 cost: Cost = 'gini',
                 cost_min: float = 0.05,
                 feature_resamples: int = 10):
        self._root: Node = Node()
        setattr(self._root, "_COST_MIN", cost_min)
        setattr(self._root, "_FEATURE_RESAMPLES", feature_resamples)

        # Random depth computation
        if str(depth) == 'random':
            self.depth = \
                np.random.choice(np.arange(1, min(len(labels) // 2, 10)), 1)[0]
        else:
            self.depth = depth

        self._cost = cost  # TODO :: make cost interchangeable in training
        self._data = data
        self._labels = labels

    def train(self):
        """Trains this decision tree"""
        self._root = self._root.split(self._data, self._labels, self.depth)
        self._root

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Traverses through tree and returns class prediction for every record"""
        predictions = []

        # In case we are passed a single numpy array with hopes of prediction
        if len(data.shape) == 1:
            data = np.array([data])

        # Appending class predictions to prediction array
        for record in data:
            predictions.append(self._root.predict_class(record))

        return predictions


