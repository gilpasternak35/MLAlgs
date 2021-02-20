from __future__ import annotations

import logging
from copy import copy, deepcopy
from typing import Any, List, Union, Tuple, Dict

import numpy as np

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
                 threshold: Union[float, int],
                 class_state: int):
        """Constructs a childless node given data and threshold"""
        self._left: Node = None
        self._right: Node = None
        self._data = data
        self._threshold = threshold
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
        return f'Node with data: {str(self._data)}'

    def _compute_threshold(self, _data: np.ndarray, _labels: np.ndarray) \
            -> Dict[str, Union[Union[float, int, np.ndarray], Any]]:
        """Updates the threshold for this node"""
        optimal_cost = np.inf
        optimal_threshold = 0
        optimal_indices_smaller = np.ndarray([])
        optimal_indices_larger = np.ndarray([])

        for feature in _data[0]:
            for i in range(self._FEATURE_RESAMPLES):
                threshold = np.random.choice(feature, size=1)
                indices_larger = np.where(feature >= threshold)
                indices_smaller = np.where(feature < threshold)
                impurity = GiniImpurity.compute(_labels[indices_larger],
                                                _labels[indices_smaller])
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
