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
        for record in data:
            predictions.append(self._root.predict_class(record))

        return predictions


if __name__ == "__main__":
    # tree = DecisionTree(data= np.array([[1,2,3,4],
    #                                     [1,3,5,6],
    #                                     [1,4,7,8],
    #                                     [10,10,10,9]]),
    #                     labels = np.array([0,0,1,1]))
    # tree.train()
    # print(tree.predict([[4, 6, 7, 9],
    #                     [1, 6, 7, 10]]))
    # Height
    data_ = np.array([[5.9, 5.0, 6.2, 5.9, 5.4, 5.3, 6.6, 5.5],
                      # Weight
                      [122.2, 105.2, 180.0, 122.2, 120.1, 110.0, 190.0, 150.1],
                      # Plays basketball?
                      [1, 0, 1, 1, 0, 0, 1, 0]])
    labels_ = np.array([1,2, 1, 2, 1, 2, 2, 2])

    tree = DecisionTree(data=data_, labels=labels_)
    tree.train()
    class_dict = {1: "man", 2: "woman"}
    pred = tree.predict([[5.7, 160, 1]])
    for val in pred:
        print(class_dict[val])
