from typing import Union

import numpy as np

from python.src.models.decision_tree.node import Node
from python.src.models.utils.cost import Cost


class DecisionTree(object):
    """Represents a randomly sampled decision tree algorithm

        todo :: maybe rename to random decision tree
    """
    _MIN_DEPTH = 3

    def __init__(self,
                 data: np.ndarray,
                 labels: np.ndarray,
                 depth: Union[str, int] = 'random',
                 cost: Cost = 'gini',
                 cost_min: float = 0.05,
                 feature_resamples: int = 10):
        # TODO :: arbitrary depth
        #      range: [3, min(len(data)//2, 10]
        self._root: Node = Node()
        setattr(self._root, "_COST_MIN", cost_min)
        setattr(self._root, "_FEATURE_RESAMPLES", feature_resamples)
        if isinstance(depth, str):
            raise ValueError("HEy You we didn't do this yet. Chill bro.")
        else:
            self._max_depth = depth

        self._cost = cost  # TODO :: make cost interchangeable in training
        self._data = data
        self._labels = labels

    def train(self):
        """Trains this decision tree"""
        self._root = self._root.split(self._data, self._labels, self._max_depth)
        return self

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Traverses through tree and returns class prediction for every record"""
        predictions = []
        for record in data:
            predictions.append(self._root.predict_class(record))

        return predictions


if __name__ == "__main__":
    tree = DecisionTree()



