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
        # TODO :: Must define a LabeledPoint.__eq__() method for this to work
        self._elements = np.array([]) if points is None else points

    @property
    def points(self) -> List[LabeledPoint]:
        return deepcopy(self._elements)

    def insert(self, vec: LabeledPoint) -> None:
        self._elements = np.append(self._elements, vec)

    def remove(self, element: LabeledPoint):
        # TODO :: numpy makes u write yourself
        pass

    def mean(self, axis: int = 0) -> np.array:
        # TODO :: gil, check me here
        return self._elements.mean(axis=axis)


