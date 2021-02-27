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
        cluster_values = model.fit(points)
        test_cluster_1 = np.array([77, 55, 68])
        test_big = np.array([1,2,3])


        self.assertEqual(SMALL, model.predict(test_small))
        self.assertEqual(BIG, model.predict(test_big))

    def test_point_cluster_2(self):
        """More complex test case"""
        import pandas as pd
        df = pd.read_csv('~/Documents/Projects/MLAlgs/SyntheticDataset.csv')
        features = df[['Weight(Pounds)']].to_numpy()
        labels = df['Gender'].to_numpy()

        model = KMeans()
        model = model.fit(features, labels)

        gender_map = {0: 'male', 1: 'female'}
        vector = namedtuple('vector', ['features', 'label'])
        jesse = vector(np.array([133]), 0)
        gil = vector(np.array([175]), 0)
        self.assertEqual(model.predict(gil.features), gil.label)
        self.assertEqual(model.predict(jesse.features), jesse.label)