from collections import Iterable
from copy import deepcopy
from typing import Iterator, List, Any

import numpy as np

# Todo: complete this
from src.utils.labeled_point import LabeledPoint


class LabeledPointList(Iterable):
    """Represents a list of labeled points"""

    def __iter__(self) -> Iterator[LabeledPoint]:
        pass

    def __init__(self, points: np.array = None, representative: Any = None):
        self._elements = np.array([]) if points is None else points
        self._representative = representative

    @property
    def points(self) -> List[LabeledPoint]:
        """Returns copy of points in list"""
        return deepcopy(self._elements)

    def get_representative(self):
        return deepcopy(self._representative)

    def set_representative(self, new_representative):
        self._representative = new_representative

    def insert(self, vec: LabeledPoint) -> None:
        """Inserts element to labeled point list"""
        self._elements = np.append(self._elements, vec)

    def remove(self, element: LabeledPoint):
        """Removes element from a labeled point list"""
        # Keeping all elements not exactly equal to point
        # Todo: is there a better way to do this? What about matching instances?
        self._elements = [point for point in self._elements if point != element]

    def vector_mean(self, axis: int = 0) -> np.array:
        """Computes and returns vector mean"""
        # Computing mean of the vector, along 0th axis
        sum_vec = [0,0,0]
        for element in self._elements:
            sum_vec = np.add(sum_vec, element.vector)
        return np.divide(sum_vec, len(self._elements))

    def __str__(self):
        return f"representative: {self._representative}, points: {[str(elem) for elem in self._elements]}"


