from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from .base_model import BaseModel
import pandas as pd

class LogRegression(BaseModel):
    _model: LogisticRegression

    def __init__(self, params: dict):
        super().__init__(params)
        self._model = LogisticRegression(**self.parameters)

    def fit(self, X_train, y_train, weight=None):
        assert len(y_train.shape) == 1, "y_train should be 1D"

        self._model.fit(X_train, y_train)

    def predict(self, X):
        return self._model.predict(X)

    def feature_importance(self):
        return self._model.coef_.flatten()

    def plot_feature_importance(self, feature_names: list[str], show_n):
        df = pd.DataFrame(columns=["feature", "importance"], data=zip(feature_names, self.feature_importance()))
        df = df.sort_values("importance", ascending=True)
        df = df[df["importance"] > 0]
        df = df.tail(show_n)

        df.plot(kind='barh', x='feature', y='importance')
        plt.show()

