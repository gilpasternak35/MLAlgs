from unittest import TestCase


import numpy as np

from src.models.unsupervised.kmeans import KMeans


class TestKNeighborsClassifier(TestCase):

    def test_predict(self):
        """Tests predict function"""
        self.test_point_cluster_1()
        self.test_point_cluster_2()

    def test_point_cluster_1(self):
        """Simple point cluster test case - should converge to 2 centroids"""
        # 0 is "small", 1 is "big"
        # This test case shows ability to separate large numbers from small
        points = np.array([
            [0, 1, 2],
            [0, 1, 1],
            [95, 95, 95],
            [95, 95, 95],
            [95, 95, 95],
            [0, 2, 4]])

        model = KMeans(k_centroids=2, distance="euclidean")
        model = model.fit(points)
        cluster_predictions = model.predict([[0, 1, 1], [94, 95, 92], [17, 12, 11]])

        self.assertEqual(cluster_predictions, np.array([[0, 1, 1.5], [95, 95, 95], [0, 1, 1.5]]))

    def test_point_cluster_2(self):
        """More complex test case"""
        points = np.array([
            [0, 1, 15, 12, 8, 4],
            [0, 1, 15, 4, 12, 2],
            [0, 0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1, 2],
            [94, 92, 57, 92, 92, 92],
            [94, 91, 52, 91, 93, 92]])

        model = KMeans(k_centroids=3, distance="euclidean")
        model = model.fit(points)
        cluster_predictions = model.predict([[0, 1, 14, 12, 7, 2], [94, 92, 52, 91, 93, 92], [[1, 0, 0, 1, 1, 2]]])

        self.assertEqual(cluster_predictions, np.array([[0, 1, 15, 8, 10, 3],[94, 91.5, 54.5, 91.5, 92.5, 92], [0, 0, 0, 1, 1, 1.5]]))