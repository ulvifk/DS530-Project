from itertools import product

import numpy as np

from src.utils import process_data
from src.problem_data import ProblemData
from sklearn.model_selection import KFold
from tqdm import tqdm


class HyperParameterSearch:
    _model_name: type
    _problem_data: ProblemData
    _n_splits: int
    _search_space: dict
    scores: list[dict, list[np.array]]
    _error_metric: callable
    avg_score: np.float64
    best_param: dict
    best_score: np.float64
    _search_space: list[dict]

    def __init__(self, model_name: type, problem_data: ProblemData, n_splits: int, error_metric: callable, search_space: dict):
        self._model_name = model_name
        self._problem_data = problem_data
        self._n_splits = n_splits
        self._error_metric = error_metric
        self._search_space = [dict(zip(search_space.keys(), values)) for values in product(*search_space.values())]
        self._search_space.append({})
        self.scores = []
        self.best_param = {}
        self.best_score = np.float64(0)

        self._search()

    def _search(self):
        p_bar = tqdm(self._search_space)
        for params in p_bar:
            p_bar.set_description(f"Testing parameters: {params}")
            p_bar.set_postfix({"Best Score":self.best_score})
            score = self._test_parameters(params)
            self.scores.append((params, score))
            if score > self.best_score:
                self.best_score = score
                self.best_params = params

    def _test_parameters(self, parameters: dict) -> np.float64:
        data_sets = list(KFold(n_splits=self._n_splits, shuffle=True)
                         .split(self._problem_data.df_X, self._problem_data.df_y))

        scores = []
        for train_indices, test_indices in data_sets:

            scores.append(self._test_and_return_score(self._model_name, train_indices, test_indices, parameters))

        return np.mean(scores)

    def _test_and_return_score(self, model_name: type, train_indices: np.array, test_indices: np.array, parameters: dict) -> np.float64:
        train_X, train_Y = (self._problem_data.df_X.values[train_indices],
                            self._problem_data.df_y.values[train_indices])
        train_Y = train_Y.reshape(-1)

        test_X, test_Y = (self._problem_data.df_X.values[test_indices],
                          self._problem_data.df_y.values[test_indices])
        test_Y = test_Y.reshape(-1)

        train_X, train_Y, test_X, test_Y = process_data(train_X, train_Y, test_X, test_Y)

        model = model_name(parameters)
        model.fit(train_X, train_Y)

        return self._error_metric(model, test_X, test_Y)
