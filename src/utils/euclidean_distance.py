import numpy as np

from src.utils.distance import Distance


class Euclidean(Distance):

    @staticmethod
    def distance(vec_1: np.array, vec_2: np.array, *args, **kwargs) \
            -> np.array:
        """
        Computes euclidean distance between vec_1 and vec_2.

        TODO :: Params
        """
        return np.sqrt(np.sum(np.power(vec_1 - vec_2, 2)))
