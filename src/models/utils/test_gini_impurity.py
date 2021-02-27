from unittest import TestCase

from python.src.models.utils.gini_impurity import GiniImpurity


class TestGiniImpurity(TestCase):
    def test_compute(self):
        test_1 = [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ]
        test_2 = [
            [1, 2]
        ]
        test_3 = [
            [2, 2, 'Bob']
        ]

        cost = GiniImpurity()

        # Test 1
        self.assertAlmostEqual(0.0, cost.compute(*test_1), delta=0.02)
        self.assertAlmostEqual(0.5, cost.compute(*test_2), delta=0.02)
        self.assertAlmostEqual(0.4555, cost.compute(*test_3), delta=0.02)

