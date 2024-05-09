import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from .base_model import BaseModel


class XGBoost(BaseModel):
    _model: XGBClassifier

    def __init__(self, parameters=None):
        super().__init__(parameters)
        self._model = XGBClassifier(**self.parameters)

    def fit(self, train_X: np.ndarray, train_Y: np.array, weights=None):
        assert len(train_Y.shape) == 1, "train_Y should be 1D"

        if weights is not None:
            self._model.fit(train_X, train_Y, sample_weight=weights)
        else:
            self._model.fit(train_X, train_Y)

    def predict(self, X: np.ndarray) -> np.array:
        return self._model.predict(X).reshape(-1)
