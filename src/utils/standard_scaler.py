import numpy as np


class StandardScaler:
    """Class for scaling data"""

    @staticmethod
    def scale(self, data: np.array) -> np.array:
        """Scales data with implicit assumption of normal distribution, returns z_scores"""
        return (data - np.mean(data)) / np.std(data)

