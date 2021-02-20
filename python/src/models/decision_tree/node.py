from __future__ import annotations

import logging
from copy import copy, deepcopy
from typing import Any, List, Union, Tuple, Dict

import numpy as np
import pandas as pd

from python.src.models.utils.gini_impurity import GiniImpurity

log = logging.getLogger(__name__)


class Node(object):
    """Represents a single node in a tree"""
    RIGHT_CLASS = 1
    LEFT_CLASS = 0
    _COST_MIN = 0.05
    _FEATURE_RESAMPLES = 10

    def __init__(self, data: np.ndarray,
                 labels: np.ndarray,
                 class_state: int = 2):   # Note: class state initialized to 2 for root node as it does not have its own
        """Constructs a childless node given data and threshold"""
        self._left: Node = None
        self._right: Node = None
        self._threshold = None
        self._data = data.T
        self._class_state = class_state
        self._labels = labels

    @property
    def data(self):
        return None if self._data is None else deepcopy(self._data)

    @property
    def labels(self):
        return None if self._labels is None else deepcopy(self._labels)

    @property
    def left(self) -> Node:
        return copy(self._left)

    # TODO :: abstract set_left and set_right w/ setattr()
    def set_left(self, child: Union[Node, List[Any]]) -> None:
        """Sets the left child of this node

            :param child: either a constructed node or data to construct a
                node with
        """
        if isinstance(child, Node):
            if self._left is not None:
                log.warn(f'Overwriting left child {self._left} of {self}')
            self._left = child
            # TODO :: replace this w/ better method once figured out
            setattr(self._left, '_class_state', self.LEFT_CLASS)
        else:
            self._left = Node(data=child, class_state=self.LEFT_CLASS)

    @property
    def right(self) -> Node:
        return copy(self._right)

    def set_right(self, child: Union[Node, List[Any]]) -> None:
        """Sets the right child of this node

            :param child: either a constructed node or data to construct a
                node with
        """
        if isinstance(child, Node):
            if self._right is not None:
                log.warn(f'Overwriting left child {self._right} of {self}')
            self._right = child
            # TODO :: replace this w/ better method once figured out
            setattr(self._right, '_class_state', self.RIGHT_CLASS)
        else:
            self._right = Node(data=child, class_state=self.RIGHT_CLASS)

    def __str__(self) -> str:
        return f'Node with data: \n{str(self._data)}\n and labels \n{str(self._labels)}\n'

    def _compute_threshold(self) \
            -> Dict[str, Union[Union[float, int, np.ndarray], Any]]:
        """Updates the threshold for this node"""
        optimal_cost = np.inf
        optimal_threshold = 0
        optimal_indices_smaller = np.ndarray([])
        optimal_indices_larger = np.ndarray([])

        for feature in self._data:
            for i in range(self._FEATURE_RESAMPLES):
                print(feature)
                threshold = np.random.choice(feature, size=1)
                indices_larger = np.where(feature >= threshold)
                indices_smaller = np.where(feature < threshold)
                impurity = GiniImpurity.compute(self._labels[indices_larger],
                                                self._labels[indices_smaller])
                if impurity < optimal_cost:
                    optimal_cost = impurity
                    optimal_threshold = threshold
                    optimal_indices_smaller = indices_smaller
                    optimal_indices_larger = indices_larger

        return {
            'cost': optimal_cost,
            'threshold': optimal_threshold,
            'smaller': optimal_indices_smaller,
            'larger': optimal_indices_larger
        }

    def split(self, remaining_depth: int, data: np.ndarray,
              labels: np.ndarray) -> Node:
        """Split nodes"""
        cost, self._threshold, smaller_indices, larger_indices = \
            self._compute_threshold(self._data)

        # for feature in self._data:
        #     feature_under_threshold = []
        #     feature_over_threshold = []
        #     for i in range(len(feature)):
        #         if i


        over_threshold = np.ndarray([])
        under_threshold = np.ndarray([])

        if remaining_depth > 0 and cost > self._COST_MIN:

            # left: TODO ::  make sure this part is actually gonna work oops
            self.set_left(self._left.split(remaining_depth - 1),
                          data=under_threshold)
            # right:
            self.set_right(self._right.split(remaining_depth - 1),
                           data=over_threshold)
        else:
            # this is a leaf representing a class. ...return?
            return self


if __name__ == "__main__":
    data = pd.DataFrame({"feature1": [1,3,3,4,5,6,7],
                         "feature2": [0,0,0,0,0,0,0],
                         "feature3": [1,1,1,1,2,2,2],
                         "feature4": [1,2,2,7,7,7,7]})
    labels = pd.Series([0,0,1,1,1,1,1])
    nd = Node(data=data.to_numpy(), labels=labels.to_numpy())
    print(nd)
    print(nd._compute_threshold(_data=data.to_numpy().T, _labels=labels.to_numpy()))

