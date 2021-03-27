from typing import Any
import numpy as np

from src.models.supervised.classifier.base_classifier import BaseClassifier
from src.utils.stat_utils import predict_nearest_class, shuffle_data


class Perceptron(BaseClassifier):
    """Perceptron Base Class - partial functionality"""

    def __init__(self, max_iter: int = 30):
        super().__init__()
        self.max_iter = max_iter
        self.weight_vector = None
        self.bias_term = None
        self._classes = None

    def fit(self, features: Any, labels: Any) -> BaseClassifier:
        """Fits a Perceptron classifier to a set of features and labels, returns Model"""
        # Tuple together features and labels
        data = np.array([(feature, label) for feature, label in zip(features, labels)])

        # Traversal through num_iterations
        for iteration in range(self.max_iter):
            # Shuffle
            shuffle_data(data)
            # Traversal through tuple-d features and labels - unpack examples one at a time\

            for feature, label in data:
                prediction = np.dot(self.weight_vector, feature) + self.bias_term
                # Check if we are correctly classifying via sign test
                if prediction * label <= 0:
                    # Update weights and bias
                    self._update_weights(feature, label)
        return self

    def predict(self, data: np.array) -> int:
        """Chooses between a set of binary classes"""
        prediction: int = np.dot(data, self.weight_vector)
        weight_prediction = predict_nearest_class(prediction, self._classes)
        return weight_prediction

    def _update_weights(self, feature: np.array, label: int):
        """Updates model weights and bias term"""
        self.bias_term += label
        self.weight_vector += label * feature

    def randomly_initialize_parameters(self, num_features: int):
        """Initializing bias to 0 and weights using uniform random number generator"""
        self.bias_term = 0
        self.weight_vector = np.array([np.random.uniform(0.001, 1) for i in range(num_features)])

    def manually_initialize_parameters(self, weight_vector: np.array, bias_term: int):
        """Manually initializing bias and weights"""
        self.weight_vector = weight_vector
        self.bias_term = bias_term
