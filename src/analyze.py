import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

from src import BaseModel, ProblemData
from src import utils

class Analyze:
    model: BaseModel
    problem_data: ProblemData
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray

    def __init__(self, model: type, problem_data: ProblemData, parameters: dict):
        self.model = model(parameters)
        self.problem_data = problem_data

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            problem_data.df_X.values,
            problem_data.df_y.values.reshape(-1),
            test_size=0.2,
            random_state=42
        )

        self.model.fit(self.X_train, self.y_train)


    def analyze(self):
        print("Accuracy: ", utils.calculate_accuracy_score(self.model, self.X_test, self.y_test))
        print("AUC: ", utils.calculate_auc_score(self.model, self.X_test, self.y_test))
        print("F1 Score: ", utils.calculate_f1_score(self.model, self.X_test, self.y_test))

        y_pred = self.model.predict(self.X_test)
        test_df = pd.DataFrame(columns=self.problem_data.df_X.columns, data=self.X_test)
        test_df["income"] = y_pred

        utils.get_summary(self.problem_data.df)
        print("################################### Test Data ###################################")
        utils.get_summary(test_df)




