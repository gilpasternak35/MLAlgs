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
    _ROOT_CLASS_STATE = 2

    def __init__(self, class_state: int = _ROOT_CLASS_STATE):
        """Constructs a childless node given data and threshold"""
        # Note: class state initialized to 2 for root as it does not have actual state
        self._left: Node = None
        self._right: Node = None
        self._threshold = None
        self._class_state = class_state

    @property
    def left(self) -> Node:
        return copy(self._left)

    # TODO :: abstract set_left and set_right w/ setattr()
    def set_left(self, child: Union[Node, Any]) -> None:
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
            self._left = Node(class_state=self.LEFT_CLASS)

    @property
    def right(self) -> Node:
        return copy(self._right)

    def set_right(self, child: Union[Node, Any]) -> None:
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
            self._right = Node(class_state=self.RIGHT_CLASS)

    def __str__(self) -> str:
        return f'Node with state: {self._class_state} and threshold {str(self._threshold)}\n'

    def _compute_threshold(self, data: np.ndarray, labels: np.ndarray) \
            -> Dict[str, Union[Union[float, int, np.ndarray], Any]]:
        """Updates the threshold for this node"""
        optimal_cost = np.inf
        optimal_threshold = 0
        optimal_indices_smaller = np.ndarray([])
        optimal_indices_larger = np.ndarray([])

        for feature in data:
            for i in range(self._FEATURE_RESAMPLES):
                print(feature)
                threshold = np.random.choice(feature, size=1)
                indices_larger = np.where(feature >= threshold)[0]  # Todo: find a more elegant way to do this
                indices_smaller = np.where(feature < threshold)[0]
                impurity = GiniImpurity.compute(labels[indices_larger],
                                                labels[indices_smaller])
                if impurity < optimal_cost:
                    optimal_cost = impurity
                    optimal_threshold = threshold
                    optimal_indices_smaller = indices_smaller
                    optimal_indices_larger = indices_larger

        return {
            'cost': optimal_cost,
            'threshold': optimal_threshold[0],
            'smaller': optimal_indices_smaller,
            'larger': optimal_indices_larger
        }

    def split(self, remaining_depth: int, data: np.ndarray,
              labels: np.ndarray) -> Node:
        """Split nodes"""

        # While we haven't reached optimal, maximum depth, or empty data
        if remaining_depth > 0 and len(labels) > 1:

            # If shape of data does not match expected label length, transposing
            if len(data[0]) != len(labels):
                data = data.T

            # If nonempty, compute thresholds for split
            cost, self._threshold, smaller_indices, larger_indices = \
                self._compute_threshold(data, labels).values()

            # If we have reached optimal value, this will be our last split
            if cost > self._COST_MIN:
                remaining_depth = 1

            # Numpy splitting along column axis
            data_over_threshold = data[:, larger_indices]
            data_under_threshold = data[:, smaller_indices]
            labels_under_threshold = labels[smaller_indices]
            labels_over_threshold = labels[larger_indices]

            # If no new_left and new_right have been initialized, building Nodes
            if self._left is None:
                self.set_left(self.LEFT_CLASS)

            if self._right is None:
                self.set_right(self.RIGHT_CLASS)

            self._left.split(remaining_depth - 1,
                             data=data_over_threshold,
                             labels=labels_over_threshold)
            # right:
            self._right.split(remaining_depth - 1,
                              data=data_under_threshold,
                              labels=labels_under_threshold)

        # Propagating back root node
        return self


if __name__ == "__main__":
    data = pd.DataFrame({"feature1": [1, 3, 3, 4, 5, 6, 7],
                         "feature2": [0, 0, 0, 0, 0, 0, 0],
                         "feature3": [1, 1, 1, 1, 2, 2, 2],
                         "feature4": [1, 2, 2, 7, 7, 7, 7]})
    labels = pd.Series([0, 0, 1, 1, 1, 1, 1])
    nd = Node(class_state=2)
    print(nd)
    tree = nd.split(data=data.to_numpy(), labels=labels.to_numpy(), remaining_depth=5)
    print(tree)
