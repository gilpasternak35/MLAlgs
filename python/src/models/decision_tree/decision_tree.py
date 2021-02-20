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
                 cost: Cost = 'gini'):
        # TODO :: arbitrary depth
        #      range: [3, min(len(data)//2, 10]
        self._root: Node = Node()
        if isinstance(depth, str):
            raise ValueError("HEy You we didn't do this yet. Chill bro.")
        else:
            self._max_depth = depth

        self._cost = cost  # TODO :: make cost interchangeable in training
        self._data = data
        self._labels = labels

    def train(self):
        """Trains this decision tree"""
        # TODO :: split nodes & preserve own structure
        #       When splitting, nodes persist w/ a threshold. thus nodes
        #       need a threshold.***
        # TODO :: calculate threshold... somewhere. Stop when depth == max
        #  OR some arbitrarily small gini impurity
        self._root.split(self._max_depth, self._data, self._labels)
        # TODO :: returns ... ?

    def predict(self, data: np.ndarray) -> np.ndarray:
        """TODO :: implement"""
        pass

