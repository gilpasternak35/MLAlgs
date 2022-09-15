import pandas as pd
import numpy as np
import statistics as stats
from logging import getLogger
from typing import Union

logger = getLogger()

class KNN_Classifier:
    
    def __init__(self, k: int):
        """
        A constructor initializing the number of examples to be used for classification
        """
        # Ensuring k is an int and odd to allow for majority, and that k > 0
        assert (isinstance(k, int) 
                and (k % 2 != 0) 
                and k > 0), "Please ensure the value of k is an even, positive integer"
        
        self.k_examples = k
        
    
    def fit(self, data: pd.DataFrame):
        """
        A Model fitting, consisting of simply storing the data
        
        @param data: A Pandas DataFrame consisting of the data the model maintains
        """
        self.data = data
        
    def __str__(self):
        """String representation of model"""
        return f"KNN Model with data: \n{data}"
    
    
    def predict(self, example: pd.Series) -> Union[int, str, bool, float]:
        """
        Classifies an example based with KNN Algorithm based on initialized k. 
        
        Explicitly assumes labels are the index of both of the series and the data
        
        @param example: An example to classify
        @returns a primitive representing the predicted label
        """
        # Warning of assumption in debug mode
        logger.warning("Warning: Assuming label is index (for simplicity of computation)")
        
        # Computing distances
        distances = (self.data.apply
                    (self.__euclidian_distance, 
                     vec2 = example, 
                     axis=1))
        
        # Sorting distances
        distances.sort_values(inplace=True)
        
        # Finding k most commmon labels
        k_most_common_labels = np.array(distances.index[0:self.k_examples])
        
        # Returning mode of array
        return stats.mode(k_most_common_labels)
        
        
    
    @staticmethod
    def __euclidian_distance(vec1: pd.Series, vec2: pd.Series) -> float:
        """l2 distance computation"""
        return np.sqrt(np.sum(np.power(vec1-vec2, 2)))
    
