from typing import Any, Union

import numpy as np

from src.models.unsupervised.unsupervised_base_classifier import UnsupervisedBaseClassifier
from src.utils.distance import Distance
from src.utils.euclidean_distance import Euclidean
from src.utils.labeled_point import LabeledPoint

class KMeans(UnsupervisedBaseClassifier):
    """An implementation of the unsupervised KMeans Clustering algorithm"""
    DISTANCES = {
        'euclidean': Euclidean.distance,
    }

    def __init__(self, k_centroids: int = 3, distance: Union[Distance, str] = 'euclidean'):
        """Instantiates a new KMeans classifier"""
        super().__init__()
        self._k_centroids = k_centroids
        self._distance = (distance if isinstance(distance, Distance)
                          else self.DISTANCES.get(distance, 'euclidean'))

    def _assign(self, point: np.array, centroids: np.array):
        # Todo: complete this tomorrow morning
        # define centroid list for each centroid
        # Keep track of which centroids have been assigned where via mutation


        # set label to label of the closest centroid




    def fit(self, features: Any) -> UnsupervisedBaseClassifier:
        """Fit the classifier to new data"""
        # Randomly initialize centroids
        centroids = np.random.choice(features, self._k_centroids)
        labeled_points = [LabeledPoint(feature, -1) for feature in features]
        has_converged = False
        # Iterate on:
        while not has_converged:

        # Assign points to centroids using Euclidean distance minimizer


            # Shift centroids to center of points that have been assigned
        # Upon convergence, return centroids (when centroids didn't move previous iteration)


    def predict(self, data) -> np.ndarray:
        """Predict on new data"""
        pass