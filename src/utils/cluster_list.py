from __future__ import annotations
import copy
from collections import Iterable
from typing import Any, Iterator
import numpy as np

from src.utils.distance import Distance
from src.utils.euclidean_distance import Euclidean
from src.utils.labeled_point import LabeledPoint
from src.utils.labeled_point_list import LabeledPointList


class ClusterList(Iterable):
    """High level class built to neatly manage a set of clusters"""
    def __init__(self, clusters: [LabeledPointList]):
        self.clusters = clusters

    def __iter__(self) -> Iterator[ClusterList]:
        for cluster in self.clusters:
            yield cluster

    def clusters(self):
        return copy.deepcopy(self.clusters)

    def representatives(self):
        return [cluster.get_representative() for cluster in self.clusters]

    def get_cluster(self, cluster_representative: Any):
        """Returns first matching instance with identical cluster representative"""
        matching = [cluster for cluster in self.clusters if all(cluster.get_representative() == cluster_representative)]
        if len(matching) > 0:
            return matching[0]
        else:
            return None

    def modify_cluster_representative(self, cluster_representative: Any, new_representative: Any):
        """Sets a new cluster representative"""
        self.get_cluster(cluster_representative).set_representative(new_representative)

    def remove_point_from_cluster(self, cluster_representative: Any, point: LabeledPoint):
        """Removes a single labeled point from a cluster"""
        chosen_cluster = self.get_cluster(cluster_representative)

        # Removing point if there
        if chosen_cluster is not None:
            chosen_cluster.remove(point)

    def add_point_to_cluster(self, cluster_representative: Any, point: LabeledPoint):
        """Adds a single labeled point to a cluster"""
        self.get_cluster(cluster_representative).insert(point)
        point.label = cluster_representative

    def compute_cluster_mean(self, cluster_representative: Any):
        """Returns the mean of a cluster"""
        cluster = self.get_cluster(cluster_representative)
        if cluster is not None:
            return cluster.vector_mean()
        else:
            return 0

    def set_cluster_representative_as_mean(self, cluster_representative: Any):
        """Sets the cluster representative to be the mean of that cluster"""
        self.modify_cluster_representative(cluster_representative, self.compute_cluster_mean(cluster_representative))

    def assign_point_to_nearest_cluster(self, point: LabeledPoint, distance: Distance = Euclidean):
        """Assigns point to nearest cluster based on distance from representative"""
        closest = {'distance': None, 'centroid': None}
        for cluster in self.clusters:
            # Compute distance to each centroid
            current_distance = point.distance(LabeledPoint(cluster.get_representative(), 0), distance)
            # keep track of closest
            if (closest.get('distance') is None
                    or current_distance < closest.get('distance')):
                closest.update({'distance': current_distance, 'centroid': cluster.get_representative()})

        # Remove from current cluster
        self.remove_point_from_cluster(point.label, point)

        # Reassign to nearest cluster
        self.add_point_to_cluster(closest['centroid'], point)

    def __str__(self):
        return [str(cluster) for cluster in self.clusters]
