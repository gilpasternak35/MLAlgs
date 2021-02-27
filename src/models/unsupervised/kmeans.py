from typing import Any, Union, Dict

import numpy as np

from src.models.unsupervised.unsupervised_base_classifier import \
    UnsupervisedBaseClassifier
from src.utils.cluster_list import ClusterList
from src.utils.distance import Distance
from src.utils.euclidean_distance import Euclidean
from src.utils.labeled_point import LabeledPoint
from src.utils.labeled_point_list import LabeledPointList


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
        self._final_cluster_list = None

    def fit(self, point_set: Any) -> UnsupervisedBaseClassifier:
        """Fit the classifier to new data"""

        # Generate centroids at random positions
        generated_centroids = [LabeledPointList(representative=vec) for vec in
                                point_set[np.random.choice(len(point_set), self._k_centroids)]]

        cluster_list = ClusterList(generated_centroids)
        has_converged = False

        # For convergence checks
        while not has_converged:
            # Assigning each point to nearest cluster
            for point in point_set:
                cluster_list.assign_point_to_nearest_cluster(LabeledPoint(point, -1))

            # Obtaining previous representatives prior to update
            previous_representatives = cluster_list.representatives

            # Reassigning clusters
            self._update_centroids(cluster_list)

            # Obtaining current representatives post update
            current_representatives = cluster_list.representatives

            # Testing for convergence
            has_converged = self._has_converged(previous_representatives, current_representatives)

        # Maintaining final_cluster_list for prediction purposes
        self._final_cluster_list = cluster_list

        # Returning classifier, whose final cluster list we can access
        return self

    def _update_centroids(self, cluster_list: ClusterList):
        """Setting all cluster centroids to be the mean of all of their assigned points"""
        for cluster in cluster_list:
            cluster_list.set_cluster_representative_as_mean(cluster.representative)

    @staticmethod
    def _has_converged(previous_centroids: np.array, current_centroids: np.array) -> bool:
        """Checks if has converged by comparing previous and current centroids"""
        # Testing positional equality for all vectors, then ensuring all vectors have all identical position
        return all([all(current == previous)
                    for current, previous in zip(previous_centroids, current_centroids)])

    def predict(self, data) -> np.ndarray:
        """Predict cluster representative of new data"""
        label_predictions = np.array([])

        # If 1D array, nesting in array so as to traverse "once" and return single prediction
        if len(data.shape) == 1:
            data = np.array([data])

        # Otherwise, traversing through vectors, creating point, and assigning
        else:
            for vector in data:
                # Building labeled point
                vector_as_point = LabeledPoint(vector, -1)

                # Computing proper centroid for labeledpoint
                self._final_cluster_list.assign_point_to_nearest_cluster(vector_as_point)

                # appending representative by obtaining new label of point
                label_predictions = np.append(label_predictions, vector_as_point.label)

        return label_predictions
