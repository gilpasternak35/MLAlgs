from __future__ import annotations

import logging
from copy import copy
from typing import Any, Union, Dict

import numpy as np
import scipy.stats as stats

from src.utils import GiniImpurity

log = logging.getLogger(__name__)


class Node(object):
    """Represents a single node in a tree"""

    _COST_MIN = 0.05
    _FEATURE_RESAMPLES = 10
    _ROOT_CLASS_STATE = 2

    def __init__(self, class_state: int = _ROOT_CLASS_STATE):
        """Constructs a childless node given data and threshold"""
        # Note: class state initialized to 2 for root as it does not have actual state
        self._left: Node = None
        self._right: Node = None
        self._threshold = None
        self._splitting_feature = None
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
        else:
            self._left = Node(class_state=child)

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
        else:
            self._right = Node(class_state=child)

    def __str__(self) -> str:
        return f'Node with state: {self._class_state} -- Threshold' \
               f' {str(self._threshold)} -- Left: {str(self._left)} --' \
               f'Right: {str(self._right)}\n'

    def _compute_threshold(self, data: np.ndarray, labels: np.ndarray) \
            -> Dict[str, Union[Union[float, int, np.ndarray], Any]]:
        """Updates the threshold for this node"""
        optimal_cost = np.inf
        optimal_threshold = 0
        optimal_feature = 0
        optimal_indices_smaller = np.ndarray([])
        optimal_indices_larger = np.ndarray([])

        for feature in data:
            for i in range(self._FEATURE_RESAMPLES):
                threshold = np.random.choice(feature, size=1)
                indices_larger = np.where(feature >= threshold)[
                    0]  # Todo: find a more elegant way to do this
                indices_smaller = np.where(feature < threshold)[0]
                impurity = GiniImpurity.compute(labels[indices_larger],
                                                labels[indices_smaller])
                if impurity < optimal_cost:
                    optimal_cost = impurity
                    optimal_feature = np.where(data == feature)[0]
                    optimal_threshold = threshold
                    optimal_indices_smaller = indices_smaller
                    optimal_indices_larger = indices_larger

        return {
            'cost': optimal_cost,
            'threshold': optimal_threshold[0],
            'splitting_feature': optimal_feature[0],
            'smaller': optimal_indices_smaller,
            'larger': optimal_indices_larger
        }

    def predict_class(self, data: Any):
        """Binary tree traversal returning class for binary tree classifier"""
        if (self._splitting_feature is None
                or data[self._splitting_feature] >= self._threshold):
            return (self._class_state if self._right is None
                    else self._right.predict_class(data))
        else:
            return self._class_state if self._left is None else self._left.predict_class(
                data)

    def split(self, data: np.ndarray,
              labels: np.ndarray,
              remaining_depth: int) -> Node:
        """Split nodes"""
        # While we haven't reached optimal, maximum depth, or empty data
        if remaining_depth > 0 and len(labels) > 1:

            # If shape of data does not match expected label length, transposing
            # print(f'data: \n{data}')
            if len(data[0]) != len(labels):
                data = data.T

            # If nonempty, compute thresholds for split
            cost, self._threshold, self._splitting_feature, smaller_indices, larger_indices = \
                self._compute_threshold(data, labels).values()

            # If we have reached optimal value, this will be our last split
            if cost <= self._COST_MIN:
                remaining_depth = 1

            # Numpy splitting along column axis
            data_over_threshold = data[:, larger_indices]
            data_under_threshold = data[:, smaller_indices]
            labels_under_threshold = labels[smaller_indices]
            labels_over_threshold = labels[larger_indices]

            # If no new_left and new_right have been initialized, building Nodes
            if self._left is None and len(labels_under_threshold > 0):
                mode = stats.mode(labels_under_threshold).mode[0]
                self.set_left(mode)
                print(f'Left Node: {self} -- Mode: {mode} -- State:'
                      f' {self._left._class_state}')
            else:
                return self

            if self._right is None and len(labels_over_threshold > 0):
                mode = stats.mode(labels_over_threshold).mode[0]
                self.set_right(mode)
            else:
                return self

            self._left.split(data=data_over_threshold,
                             labels=labels_over_threshold,
                             remaining_depth=remaining_depth - 1)
            # right:
            self._right.split(data=data_under_threshold,
                              labels=labels_under_threshold,
                              remaining_depth=remaining_depth - 1)

        # Propagating back root node
        print(f'Node getting returned: {self} -- Left: {str(self._left)} -- '
              f'Right: {str(self._right)}')
        return self
