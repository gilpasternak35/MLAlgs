from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from src.utils.distance import Distance


class LabeledPoint:
    """Represents a point in Euclidean / multi-dimensional space"""

    def __init__(self, vector: np.array, label: np.array):
        self._vector = vector
        self._label = label

    def __eq__(self, o: object) -> bool:
        return (isinstance(o, LabeledPoint)
                and self.vector == o.vector
                and self.label == o.label)

    @property
    def vector(self) -> np.array:
        """Returns a copy of this point's vector.
        NOTE :: Changes made to the copy are not reflected by the original
        object.
        """
        return deepcopy(self._vector)

    @vector.setter
    def vector(self, new_vec: np.array) -> None:
        """Setter for vector"""
        self._vector = new_vec

    @property
    def label(self) -> np.array:
        """Returns a copy of this point's label.
        NOTE :: Changes made to the copy are not reflected by the original
        object.
        """
        return deepcopy(self._label)

    @label.setter
    def label(self, new_label: Any):
        """Setter for label"""
        self._label = new_label

    def distance(self, other_point: LabeledPoint, distance: Distance) -> float:
        """Distance computation between two labeled points"""
        return distance.distance(self.vector, other_point.vector)

    def __str__(self):
        """String representation of a labeled point"""
        return f"Vector: {self.vector}, label: {self._label}"
