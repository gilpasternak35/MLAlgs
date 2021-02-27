import copy
from typing import Any
import numpy as np

from src.utils.labeled_point import LabeledPoint
from src.utils.labeled_point_list import LabeledPointList


class ClusterList:
    """High level class built to neatly manage a set of clusters"""
    def __init__(self, clusters: [LabeledPointList]):
        self.clusters = np.array(clusters)

    @property
    def clusters(self):
        return copy.deepcopy(self.clusters)

    def get_cluster(self, cluster_representative: Any):
        """Returns first matching instance with identical cluster representative"""
        return [cluster for cluster in self.clusters if cluster.representative == cluster_representative][0]

    def modify_cluster_representative(self, cluster_representative: Any, new_representative: Any):
        """Sets a new cluster representative"""
        self.get_cluster(cluster_representative).set_representative(new_representative)

    def remove_point_from_cluster(self, cluster_representative: Any, point: LabeledPoint):
        """Removes a single labeled point from a cluster"""
        self.get_cluster(cluster_representative).remove(point)

    def add_point_to_cluster(self, cluster_representative: Any, point: LabeledPoint):
        """Adds a single labeled point to a cluster"""
        self.get_cluster(cluster_representative).insert(point)

    def compute_cluster_mean(self, cluster_representative: Any):
        """Returns the mean of a cluster"""
        return self.get_cluster(cluster_representative).mean()

    def set_cluster_representative_as_mean(self, cluster_representative):
        """Sets the cluster representative to be the mean of that cluster"""
        self.modify_cluster_representative(cluster_representative, self.compute_cluster_mean(cluster_representative))
