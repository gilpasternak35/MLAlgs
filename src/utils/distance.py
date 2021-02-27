from abc import ABC, abstractmethod
from typing import Any


class Distance(ABC):
    """Represents a distance function object."""

    @staticmethod
    @abstractmethod
    def distance(self, vec_1: Any, vec_2: Any, *args, **kwargs):
        pass
