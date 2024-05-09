import numpy as np

from src.utils import process_data
from src.problem_data import ProblemData
from sklearn.model_selection import KFold
from tqdm import tqdm


class KFoldCrossValidation:
    _model_name: type
    _problem_data: ProblemData
    _n_splits: int
    scores: list[np.array]
    _error_metric: callable
    avg_score: np.float64

    def __init__(self, model_name: type, problem_data: ProblemData, n_splits: int, error_metric: callable):
        self._model_name = model_name
        self._problem_data = problem_data
        self._n_splits = n_splits
        self._error_metric = error_metric
        self.scores = {}

        self._cross_validate()

    def _cross_validate(self):
        data_sets = list(KFold(n_splits=self._n_splits, shuffle=True, random_state=0)
                         .split(self._problem_data.df_X, self._problem_data.df_y))

        self.scores = []
        for train_indices, test_indices in tqdm(data_sets):
            self.scores.append(self._test_and_return_score(self._model_name, train_indices, test_indices))

        self.avg_score = np.mean(self.scores)

    def _test_and_return_score(self, model_name: type, train_indices: np.array, test_indices: np.array) -> np.float64:
        train_X, train_Y = (self._problem_data.df_X.values[train_indices],
                            self._problem_data.df_y.values[train_indices])
        train_Y = train_Y.reshape(-1)

        test_X, test_Y = (self._problem_data.df_X.values[test_indices],
                          self._problem_data.df_y.values[test_indices])
        test_Y = test_Y.reshape(-1)

        train_X, train_Y, test_X, test_Y = process_data(train_X, train_Y, test_X, test_Y)

        model = model_name()
        model.fit(train_X, train_Y)

        return self._error_metric(model, test_X, test_Y)
