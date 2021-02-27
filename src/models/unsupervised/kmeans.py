from typing import Any, Union, Dict

import numpy as np

from src.models.unsupervised.unsupervised_base_classifier import \
    UnsupervisedBaseClassifier
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
        self._final_centroids = None

    def _assign(self, point: LabeledPoint, centroids: np.array, centroid_cluster_list: LabeledPointList):
        """Assigns a labeled point to a cluster"""
        # Todo: complete this tomorrow morning
        # define centroid list for each centroid
        # Keep track of which centroids have been assigned where via mutation
        closest = {'distance': None, 'centroid': None}
        for centroid in centroids:
            # Compute distance to each centroid
            distance = point.distance(centroid, self._distance)
            # keep track of closest
            if (closest.get('distance') is None
                    or distance < closest.get('distance')):
                closest.update({'distance': distance, 'centroid': centroid})
        # set label to label of the closest centroid

        point.label = closest.get('centroid').label

    def fit(self, point_set: Any) -> UnsupervisedBaseClassifier:
        """Fit the classifier to new data"""
        # Data structure. Centroid labels are keys. Index by centroid,
        # then ['centroid'] or ['cluster'] to get relevant data
        # { label:
        #         {
        #         'centroid': LabeledPoint,
        #         'cluster': LabeledPointList,
        #     }
        # }

        # Randomly initialize centroids as LabeledPoints
        # Builds a LabeledPointList named centroids
        generated_centroids = [LabeledPointList(representative=vec) for vec in
                               point_set[np.random.choice(len(point_set), self._k_centroids)]]

        # Each centroid will have a cluster_list, represented by the centroid with data points as the elements
        centroid_cluster_list = np.array(generated_centroids)

        feature_points = [LabeledPoint(feature, -1) for feature in point_set]
        has_converged = False

        # Iterate on:
        while not has_converged:
            # Building representative list
            current_centroids = [cluster.representative for cluster in centroid_cluster_list]

            # Assign points to centroids using Euclidean distance minimizer
            for feature in feature_points:
                self._assign(feature, current_centroids, centroid_cluster_list)
                centroid_means[feature.label] += feature.vector

            # computing mean and assigning to computed_centroid_means dictionary
            computed_centroid_means = {
                k: vector_sum.mean()
                for k, vector_sum in centroid_means.items()
            }

            # Shift centroids to center of points that have been assigned
            last_centroids = {point.label: point.vector for point
                              in current_centroids}
            self._update_centroids(current_centroids, computed_centroid_means)

            # Since algorithm guaranteed to converge
            has_converged = self._has_converged(last_centroids, data)

        # Upon convergence, return centroids (when centroids didn't move
        # previous iteration)
        self._final_centroids = current_centroids

        # Returning centroids
        return enumerate(map(lambda k: data[k].get('centroid'), data))

    def _update_centroids(self, centroids, computed_centroid_means):
        for centroid in centroids:
            mean_vector = computed_centroid_means.get(centroid.label)
            if mean_vector is None:
                raise RuntimeError(f"Centroid counts do not match existing "
                                   f"centroids. Existing: {centroids}"
                                   f"Counts: {computed_centroid_means}")
            else:
                centroid.vector = mean_vector

    @staticmethod
    def _has_converged(last_centroids, data) -> bool:
        converged = True
        for last, new in zip(last_centroids, [k['centroid']
                                              for _, k in data.items()]):
            converged = converged and (last == new)
        return converged

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
                self.assign(vector_as_point, self._final_centroids)

                # appending to label_predictions
                label_predictions = np.append(label_predictions, vector_as_point.label)

        return label_predictions









