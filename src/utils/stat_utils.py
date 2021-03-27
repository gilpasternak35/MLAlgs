import collections
from typing import Iterable, Any, List
import numpy as np


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


def vector_norms(vectors: np.array):
    """Computes a list of vector norms for a list of vectors"""
    norms = np.array([])
    for vector in vectors:
        norms = np.append(vector_norm(vector))
    return norms


def vector_norm(vector: np.array):
    """Computes vector norm for an arbitrary vector"""
    return np.divide(vector, np.sum(np.power(vector, 2)))


def shuffle_data(input_vectors: np.ndarray):
    """
    Shuffles the order of an array of input vectors
    Note: This should be a list of input vectors
    """
    np.random.shuffle(input_vectors)


def predict_nearest_class(value: int, classes: np.array) -> int:
    """Predicts nearest class from a list of classes"""
    curr_min = np.inf
    final_class = None
    for chosen_class in classes:
        new_dist = abs(value - chosen_class)
        if new_dist < curr_min:
            final_class = chosen_class
            curr_min = new_dist
           
    return final_class

            


