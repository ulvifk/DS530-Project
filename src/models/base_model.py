from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

class BaseModel:
    parameters: dict

    def __init__(self, parameters: dict = None):
        if parameters is not None:
            self.parameters = parameters
        else:
            self.parameters = {}

    @abstractmethod
    def fit(self, train_X: np.ndarray, train_Y: np.array, weights=None) -> None:
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.array:
        pass