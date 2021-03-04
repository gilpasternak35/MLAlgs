import numpy as np


def normalize_vectors(vectors: np.ndarray):
    """Normalizes a list of given vectors"""
    norms = np.array([])
    for vector in vectors:
        norms = np.append(normalize_vector(vector))
    return norms


def normalize_vector(vector: np.array):
    """Normalizes vector for an arbitrary given vector"""
    return np.divide(vector, np.sqrt(np.sum(np.power(vector, 2))))


def vector_norm(vector: np.array):
    """Computes vector norm for a given vector"""
    return np.sqrt(np.sum(np.power(vector, 2)))


def dot_product(vector_1: np.array, vector_2: np.array):
    """Computes the dot product of 2 vectors"""
    return sum(vector_1 * vector_2)