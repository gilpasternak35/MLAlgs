from typing import List, Any

from src.utils.cost import Cost


class GiniImpurity(Cost):
    """Represents Gini Impurity as a cost metric"""

    @staticmethod
    def _value_counts(node: List[Any]) -> List[int]:
        """Computes counts of unique values in a list"""
        counts = {}

        for val in node:
            counts[val] = counts.get(val, 0) + 1

        return list(counts.values())

    @staticmethod
    def _compute_impurity(counts: List[int]) -> float:
        """Computes gini impurity score from value counts"""
        length = sum(counts)
        sum_ = 0

        for count in counts:
            sum_ += ((count * 1.0) / length) ** 2

        return 1 - sum_

    @staticmethod
    def compute(*labels, **kwargs) -> float:
        """Computes GiniImpurity"""
        impurity_sum = 0
        total_length = sum(map(len, labels))

        for label in labels:
            counts = GiniImpurity._value_counts(label)
            impurity = GiniImpurity._compute_impurity(counts)
            impurity_sum += impurity * (len(label) / total_length)

        return impurity_sum
