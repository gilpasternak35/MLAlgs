from src.utils.activations.activation import Activation
import numpy as np


class Sigmoid(Activation):
    """Class representing Sigmoid activation function"""
    def compute(self, data: np.array):
        pass

    def compute_derivative(self, data: np.array):
        pass


class Tanh(Activation):
    def compute(self, data: np.array):
        pass

    def compute_derivative(self, data: np.array):
        pass


class ReLu(Activation):
    def compute(self, data: np.array):
        pass

    def compute_derivative(self, data: np.array):
        pass


class LeakyReLu(Activation):
    def compute(self, data: np.array):
        pass

    def compute_derivative(self, data: np.array):
        pass


class Softmax(Activation):
    def compute(self, data: np.array):
        pass

    def compute_derivative(self, data: np.array):
        pass