import collections
from typing import Iterable, Any, List


def value_counts(values: Iterable[Any]) -> List[int]:
    """Computes counts of unique values in a list"""
    counts = {}

    for val in values:
        counts[val] = counts.get(val, 0) + 1

    return list(counts.items())


def mode(values: Iterable[Any], axis: int = 0):
    """
    Computes the most frequently occurring value in an interable.

    """
    on_axis = [value[axis] if isinstance(value, collections.Iterable)
               else value for value in values]
    return max(value_counts(on_axis), key=lambda count: count[-1])[0]

