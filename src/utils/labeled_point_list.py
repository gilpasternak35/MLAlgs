from collections import Iterable
from copy import deepcopy
from typing import Iterator, List

import numpy as np

# Todo: complete this
from src.utils.labeled_point import LabeledPoint


class LabeledPointList(Iterable):
    """Represents a list of labeled points"""

    def __iter__(self) -> Iterator[LabeledPoint]:
        pass

    def __init__(self, points: np.array = None):
        self._elements = np.array([]) if points is None else points

    @property
    def points(self) -> List[LabeledPoint]:
        """Returns copy of points in list"""
        return deepcopy(self._elements)

    def insert(self, vec: LabeledPoint) -> None:
        """Inserts element to labeled point list"""
        self._elements = np.append(self._elements, vec)

    def remove(self, element: LabeledPoint):
        """Removes element from a labeled point list"""
        # Deleting point from array
        np.delete(self._elements, element)

    def vector_mean(self, axis: int = 0) -> np.array:
        """Computes and returns vector mean"""
        # Computing mean of the vector, along 0th axis
        return np.mean(self._elements, axis=0)


