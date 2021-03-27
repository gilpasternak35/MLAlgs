from collections import namedtuple
from unittest import TestCase


import numpy as np

from src.models.nn.perceptron.perceptron import Perceptron
from src.utils.standard_scaler import StandardScaler


class TestPerceptronClassifierClassifier(TestCase):
    """Test methods for the Perceptron Classifier"""

    def test_predict(self):
        """Tests prediction for Perceptron"""
        self._case_1()
        self._gathered_data_test()

    def _case_1(self):
        """Simple test case"""
        SMALL = -1
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
        features = np.array([StandardScaler.scale(feature) for feature in features])
        labels = np.array([SMALL,
                           SMALL,
                           BIG,
                           BIG,
                           BIG,
                           SMALL])
        test_small = np.array([-1, -3, -2])
        test_big = np.array([88, 85, 90])

        model = Perceptron(max_iter=10)
        model = model.fit(features, labels)

        self.assertEqual(SMALL, model.predict(test_small))
        self.assertEqual(BIG, model.predict(test_big))

    def _gathered_data_test(self):
        """More complex test case"""
        import pandas as pd
        label_col = ['sweet_tooth']
        feature_cols = ['height_inches', 'weekly_excercise_hours', 
                     'like_chocolate_chip_cookies', 'work_hours_weekly', 'netflix_hours_weekly']
        df = pd.read_csv('~/Documents/MlAlgs/ml_model_survey_new.csv')
        df = df[feature_cols + label_col]
        df = df.replace({'Yes': 1, 'No': -1, 'N.A.': np.nan})
        df = df.applymap(float)
        df = df.dropna()

        # Raw = True indicates the data will be passed as a numpy array
        features = df[feature_cols].apply(StandardScaler.scale, axis=0, raw=True).to_numpy()
        labels = df[label_col].to_numpy()

        model = Perceptron(max_iter=30)
        model = model.fit(features, labels)

        vector = namedtuple('vector', ['features', 'label'])
        jesse = vector(np.array([63, 3, 1, 50, 20.0]), 1)
        gil = vector(np.array([74, 12, 1, 90, 0]), 1)
        self.assertEqual(model.predict(gil.features), gil.label)
        self.assertEqual(model.predict(jesse.features), jesse.label)
