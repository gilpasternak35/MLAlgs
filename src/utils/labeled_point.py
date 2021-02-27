from __future__ import annotations
from typing import Any

import numpy as np

from src.utils.distance import Distance


class LabeledPoint:
    """Represents a point in Euclidean / multi-dimensional space"""
    def __init__(self, vector: np.array, label: Any):
        self._vector = vector
        self._label = label

    @property
    def vector(self) -> np.array:
        return self._vector

    @property
    def label(self) -> np.array:
        return self._label

    @label.setter
    def update_label(self, new_label: Any):
        self._label = new_label

    def distance(self, other_point: LabeledPoint, distance: Distance) -> float:
        return distance.distance(self.vector, other_point.vector)
