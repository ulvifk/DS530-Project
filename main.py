from src import *
from sklearn.model_selection import train_test_split

xgboost_search_space = {
    "eta": [1e-2, 1e-1, 2e-1],
    "max_depth": [3, 6, 9],
    "subsample": [0.5, 0.75, 1],
    "n_estimators": [10, 100, 200],
}


def main():
    # run_optuna()

    best_params = {'max_depth': 5, 'min_child_weight': 0.000189172108473086,
                   'subsample': 0.9409269252166749, 'colsample_bytree': 0.8574682295845758,
                   'learning_rate': 0.05215556356661998, 'n_estimators': 970}

    problem_data = ProblemData(["sex", "race", "marital-status"])
    X_train, X_test, y_train, y_test = train_test_split(problem_data.df_X.values, problem_data.df_y.values.reshape(-1), test_size=0.2)

    model = XGBoost(best_params)
    model.fit(X_train, y_train)

    global_surrogate = DecisionTreeSurrogate(model, {
        "max_depth": 3
    })
    global_surrogate.fit(X_train, y_train)
    global_surrogate.print_decision_paths(problem_data.df_X.columns)



if __name__ == "__main__":
    main()
