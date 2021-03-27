from typing import Any

import numpy as np


class StandardScaler:
    """Class for scaling data"""

    @staticmethod
    def scale(data: Any) -> Any:
        """Scales data with implicit assumption of normal distribution, returns z_scores"""
        return (data - np.mean(data)) / np.std(data)

