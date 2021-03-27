from collections import namedtuple
from unittest import TestCase

import numpy as np

from src.models.nn.perceptron.perceptron import Perceptron


class TestPerceptronClassifierClassifier(TestCase):
    """Test methods for the Perceptron Classifier"""

    def test_predict(self):
        """Tests prediction for Perceptron"""
        self._case_1()
        self._gathered_data_test()

    def _case_1(self):
        """Simple test case"""
        SMALL = 0
        BIG = 1
        # 0 is "small", 1 is "big"
        # This test case shows ability to separate large numbers from small
        features = np.array([
            [0, 1, 2],
            [0, 1, 1],
            [90, 95, 94],
            [96, 97, 94],
            [96, 95, 92],
            [0, 2, 4]])
        labels = np.array([SMALL,
                           SMALL,
                           BIG,
                           BIG,
                           BIG,
                           SMALL])
        test_small = np.array([12, 5, 8])
        test_big = np.array([88, 85, 90])

        model = Perceptron(max_iter=10)
        model = model.fit(features, labels)

        self.assertEqual(SMALL, model.predict(test_small))
        self.assertEqual(BIG, model.predict(test_big))

    def _gathered_data_test(self):
        """More complex test case"""
        import pandas as pd
        df = pd.read_csv('~/Documents/Projects/MLAlgs/ml_model_survey.xlsx')
        features = df[['Weight(Pounds)']].to_numpy()
        labels = df['Gender'].to_numpy()

        model = Perceptron()
        model = model.fit(features, labels)

        gender_map = {0: 'male', 1: 'female'}
        vector = namedtuple('vector', ['features', 'label'])
        jesse = vector(np.array([133]), 0)
        gil = vector(np.array([175]), 0)
        self.assertEqual(model.predict(gil.features), gil.label)
        self.assertEqual(model.predict(jesse.features), jesse.label)
