from unittest import TestCase

import numpy as np
import pandas as pd
from src.models.supervised.classifier.decision_tree.decision_tree import DecisionTree


class TestDecisionTree(TestCase):

    def test_decision_tree_predict_case(self):
        """Tests ability to predict in trivial case"""
        tree = DecisionTree(data=np.array([[1, 2, 3, 4],
                                           [1, 3, 5, 6],
                                           [1, 4, 7, 8],
                                           [10, 10, 10, 9]]),
                            labels=np.array([0, 0, 1, 1]))
        tree.train()
        assert tree.predict(np.array([[4, 6, 7, 9],
                            [1, 6, 7, 10]])) == [1, 0]

    def test_decision_tree_predict_height_test_case(self):
        """Test decision tree using gender classification given height + weight (data built at random)"""
        # Height
        data_ = np.array([[5.9, 5.0, 6.2, 5.9, 5.4, 5.3, 6.6, 5.5],
                          # Weight
                          [122.2, 105.2, 180.0, 122.2, 120.1, 110.0, 190.0, 150.1],
                          # Plays basketball?
                          [1, 0, 1, 1, 0, 0, 1, 0]])
        labels_ = np.array([1, 2, 1, 2, 1, 2, 2, 2])

        tree = DecisionTree(data=data_, labels=labels_)
        tree.train()
        class_dict = {1: "man", 2: "woman"}
        pred = tree.predict(np.array([[5.7, 160, 1]]))
        for val in pred:
            assert (class_dict[val] == 'man')

    def test_decision_tree_predict_synth_data_case(self):
        """Tests decision tree utilizing synthetic data set"""
        dataset = '/Users/concord/Documents/Projects/MLAlgs/SyntheticDataset.csv'
        df = pd.read_csv(dataset)

        labels = df['Gender'].to_numpy()
        features = df[['Height(Inches)', 'Weight(Pounds)']].to_numpy()

        tree = DecisionTree(data=features, labels=labels)
        tree.train()
        pred = tree.predict(np.array([[65, 180]]))

        assert pred == 1, "Prediction is false"