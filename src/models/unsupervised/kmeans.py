from typing import Any, Union, Dict

import numpy as np

from src.models.unsupervised.unsupervised_base_classifier import \
    UnsupervisedBaseClassifier
from src.utils.cluster_list import ClusterList
from src.utils.distance import Distance
from src.utils.euclidean_distance import Euclidean
from src.utils.labeled_point import LabeledPoint
from src.utils.labeled_point_list import LabeledPointList

# Todo: complete the fit method
class KMeans(UnsupervisedBaseClassifier):
    """An implementation of the unsupervised KMeans Clustering algorithm"""
    DISTANCES = {
        'euclidean': Euclidean.distance,
    }

    def __init__(self, k_centroids: int = 3,
                 distance: Union[Distance, str] = 'euclidean'):
        """Instantiates a new KMeans classifier"""
        super().__init__()
        self._k_centroids = k_centroids
        self._distance = (distance if isinstance(distance, Distance)
                          else self.DISTANCES.get(distance, 'euclidean'))

        # We should store this so that trained model may predict centroid assignment
        self._final_centroids = None

    def fit(self, point_set: Any) -> UnsupervisedBaseClassifier:
        """Fit the classifier to new data"""
        # Randomly initialize centroids as LabeledPoints
        # Builds a LabeledPointList named centroids
        generated_centroids = [LabeledPointList(representative=vec) for vec in
                                point_set[np.random.choice(len(point_set), self._k_centroids)]]
        cluster_list = ClusterList(generated_centroids)

        # For convergence checks
        previous_cluster_representatives = np.array([])

        return 0

    def _update_centroids(self, cluster_list: ClusterList):
        """Setting all cluster centroids to be the mean of all of their assigned points"""
        for cluster in cluster_list:
            cluster_list.set_cluster_representative_as_mean(cluster.representative)

    @staticmethod
    def _has_converged(last_centroids: np.array, current_centroids: np.array) -> bool:
        """Checks if has converged by comparing previous and current centroids"""
        has_converged = True
        for prev_centroid, cur_centroid in zip(last_centroids, current_centroids):
            has_converged = has_converged and prev_centroid == cur_centroid

            # Returning immediately if false
            if not has_converged:
                return has_converged

        return has_converged


    def predict(self, data) -> np.ndarray:
        """Predict on new data"""
        label_predictions = np.array([])

        # If 1D array, nesting in array so as to traverse "once" and return single prediction
        if len(data.shape) == 1:
            data = np.array([data])

        # Otherwise, traversing through vectors, creating point, and assigning
        else:
            for vector in data:
                # Building labeled point
                vector_as_point = LabeledPoint(vector, -1)

                # Assigning vector to centroid
                # Todo: update to cluster methodology - store the representative
                self.assign(vector_as_point, self._final_centroids)

                # appending to label_predictions
                label_predictions = np.append(label_predictions, vector_as_point.label)

        return label_predictions









