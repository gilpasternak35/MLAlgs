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

    def _assign(self, point: LabeledPoint, centroids: np.array):
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

    def fit(self, features: Any) -> UnsupervisedBaseClassifier:
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
        centroid_vectors = [LabeledPoint(vec, label) for vec, label in
                            enumerate(np.random.choice(features,
                                                       self._k_centroids))]
        data = {
            k: {
                'centroid': centroid_vectors[k],
                'cluster': LabeledPointList(),
            }
            for k in range(self._k_centroids)
        }

        feature_points = [LabeledPoint(feature, -1) for feature in features]
        has_converged = False
        # Iterate on:
        while not has_converged:
            # TODO :: This is a mapping from cluster labels to an array.
            #   Each time a vector is added to a cluster, sum it w/ the
            #   running total of vectors in that cluster. Average the final
            #   number.
            #   BENEFIT :: We compute mean(cluster) in the same iteration as
            #      assign(data), rather than an additional O(n*k) loop to
            #       average each cluster
            centroid_means: Dict[int, np.array] = {
                k: np.array() for k in range(self._k_centroids)}
            current_centroids = [data[k].get('centroid') for k in data]

            # Assign points to centroids using Euclidean distance minimizer
            for feature in feature_points:
                self._assign(feature, current_centroids)
                centroid_means[feature.label] += feature.vector

            # TODO :: a more accurate/correct cluster mean @Gil
            computed_centroid_means = {
                k: vector_sum.mean()
                for k, vector_sum in centroid_means.items()
            }

            # Shift centroids to center of points that have been assigned
            last_centroids = {point.label: point.vector for point
                              in current_centroids}
            self._update_centroids(current_centroids, computed_centroid_means)
            # TODO :: Less exact convergence?
            has_converged = self._has_converged(last_centroids, data)

        # Upon convergence, return centroids (when centroids didn't move
        # previous iteration)
        # TODO :: output type is open to change. This just seemed close enough.
        #       @Gil --jesse
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
        pass




