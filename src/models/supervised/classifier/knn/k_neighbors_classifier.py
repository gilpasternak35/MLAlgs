from __future__ import annotations

from typing import Union

import numpy as np

from src.models.supervised.classifier.base_classifier import BaseClassifier
from src.utils import stat_utils
from src.utils.distance import Distance
from src.utils.euclidean_distance import Euclidean


class KNeighborsClassifier(BaseClassifier):
    """A k-Nearest Neighbors classifier.

    INVARIANT: self._neighbors % 2 != 0
    """
    DISTANCES = {
        'euclidean': Euclidean.distance,
    }

    def __init__(self, k: int = 5, distance: Union[str,
                                                   Distance] = 'euclidean'):
        """Initializes a new KNeighborsClassifier with k=`k` and given distance
        metric
        TODO :: params, raises
        """
        # TODO :: Check that k is not a bad value
        if k % 2 == 0:
            raise ValueError(f'Even values of k ({k}) do not allow majority '
                             f'rule.')
        self._neighbors = k
        self._distance = (distance if isinstance(distance, Distance)
                          else self.DISTANCES.get(distance, 'euclidean'))

        self._features = None
        self._labels = None

    def fit(self, features: np.ndarray, labels: np.ndarray) \
            -> KNeighborsClassifier:
        """Save instance data"""
        # TODO :: if problems during prediction, check here to see if labels
        #  and data are mapped incorrectly
        assert features.shape[0] == labels.shape[-1]
        self._features = features
        self._labels = labels
        return self

    def predict(self, data: np.ndarray):
        # Feature space R^D
        # Given N features of shape D
        # For each feature, compute dist(feature,
        # If columns match, number of features match
        assert data.shape[-1] == self._features.shape[-1]

        if len(data.shape) == 1:
           data = np.array([data.tolist()])

        predictions = np.array([])
        for record in data:
            # 1: Compute distance between this record (1 vector of D
            #     features) and every training feature. Store.
            distances = [(self._distance(record, feature), label)
                         for feature, label in zip(self._features,
                                                   self._labels)]
            # 2: Sort by distance.
            distances.sort(key=lambda dist_tup: dist_tup[0])

            # 3: Take majority vote of k-closest
            predictions = np.append(predictions,
                                    stat_utils.mode(
                                        distances[:self._neighbors], axis=1))

        return (int(predictions[0]) if len(predictions) == 1
                else list(map(int, predictions)))

