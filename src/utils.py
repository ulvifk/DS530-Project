import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler

from src.models.base_model import BaseModel
from src.user_settings import UserSettings


def calculate_accuracy_score(model: BaseModel, X_test: np.ndarray, y_test: np.array) -> float:
    assert len(y_test.shape) == 1, "y_test should be 1D"

    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

def calculate_tnr_score(model: BaseModel, X_test: np.ndarray, y_test: np.array) -> float:
    assert len(y_test.shape) == 1, "y_test should be 1D"

    y_pred = model.predict(X_test)
    return recall_score(y_test, y_pred, pos_label=0)

def calculate_tpr_score(model: BaseModel, X_test: np.ndarray, y_test: np.array) -> float:
    assert len(y_test.shape) == 1, "y_test should be 1D"

    y_pred = model.predict(X_test)
    return precision_score(y_test, y_pred, zero_division=0)


def calculate_auc_score(model: BaseModel, X_test: np.ndarray, y_test: np.array) -> float:
    assert len(y_test.shape) == 1, "y_test should be 1D"

    y_pred = model.predict(X_test)
    return roc_auc_score(y_test, y_pred)

def calculate_f1_score(model: BaseModel, X_test: np.ndarray, y_test: np.array) -> float:
    assert len(y_test.shape) == 1, "y_test should be 1D"

    y_pred = model.predict(X_test)
    return f1_score(y_test, y_pred)


def oversample(X: np.ndarray, y: np.array) -> tuple[np.ndarray, np.array]:
    smote = SMOTE()
    return smote.fit_resample(X, y)


def normalize(train: np.ndarray, test: np.ndarray, scaler: type) -> tuple[np.ndarray, np.ndarray]:
    scaler = scaler()
    train = scaler.fit_transform(train)
    test = scaler.transform(test)
    return train, test


def process_data(train_X: np.ndarray, train_Y: np.array, test_X: np.ndarray, test_Y: np.array) -> tuple[
    np.ndarray, np.array, np.ndarray, np.array]:

    train_X, test_X = normalize(train_X, test_X, UserSettings.scaler)
    if UserSettings.is_smote:
        train_X, train_Y = oversample(train_X, train_Y)

    return train_X, train_Y, test_X, test_Y

def calculate_parity_score(model: BaseModel, test_X_df: pd.DataFrame, test_y_df: pd.DataFrame, df_original: pd.DataFrame) -> float:
    epsilon = 0.01

    dfs_races = {}
    dfs_sexes = {}
    male_indices = np.intersect1d(df_original[df_original["sex"] == "Male"].index, test_X_df.index)
    female_indices = np.intersect1d(df_original[df_original["sex"] != "Male"].index, test_X_df.index)

    white_indices = np.intersect1d(df_original[df_original["race"] == "White"].index, test_X_df.index)
    other_indices = np.intersect1d(df_original[df_original["race"] != "White"].index, test_X_df.index)

    dfs_sexes["Male"] = (test_X_df.loc[male_indices], test_y_df.loc[male_indices])

    dfs_sexes["Female"] = (test_X_df.loc[female_indices], test_y_df.loc[female_indices])

    dfs_races["White"] = (test_X_df.loc[white_indices], test_y_df.loc[white_indices])

    dfs_races["Other"] = (test_X_df.loc[other_indices], test_y_df.loc[other_indices])

    tpr_male = calculate_tpr_score(model, dfs_sexes["Male"][0].values, dfs_sexes["Male"][1].values.reshape(-1))
    tpr_female = calculate_tpr_score(model, dfs_sexes["Female"][0].values, dfs_sexes["Female"][1].values.reshape(-1))
    sex_parity = (max(np.abs(tpr_male - tpr_female) - epsilon, 0) + 1 ) ** 2

    tpr_white = calculate_tpr_score(model, dfs_races["White"][0].values, dfs_races["White"][1].values.reshape(-1))
    tpr_other = calculate_tpr_score(model, dfs_races["Other"][0].values, dfs_races["Other"][1].values.reshape(-1))
    race_parity = (max(np.abs(tpr_white - tpr_other) - epsilon, 0) + 1) ** 2

    f1_score = calculate_f1_score(model, test_X_df.values, test_y_df.values.reshape(-1))
    return f1_score - 0.3 * (sex_parity + race_parity - 2)


def get_summary(df: pd.DataFrame):
    print("Income")
    number_of_high_income = len(df[df['income'] == 1])
    print(f"Number of high income: {number_of_high_income}")
    number_of_low_income = len(df[df['income'] == 0])
    print(f"Number of low income: {number_of_low_income}")

    print("Gender")
    number_of_male = len(df[df['sex_Male'] == 1])
    print(f"Number of Male: {number_of_male}")
    number_of_female = len(df[df['sex_Female'] == 1])
    print(f"Number of Female: {number_of_female}")
    print()

    print("Race")
    number_of_white = len(df[df['race_White'] == 1])
    print(f"Number of white: {number_of_white}")
    number_of_black = len(df[df['race_Black'] == 1])
    print(f"Number of black: {number_of_black}")
    number_of_amer_indian_eskimo = len(df[df['race_Amer-Indian-Eskimo'] == 1])
    print(f"Number of Amer-Indian-Eskimo: {number_of_amer_indian_eskimo}")
    number_of_asian_pac_islander = len(df[df['race_Asian-Pac-Islander'] == 1])
    print(f"Number of Asian-Pac-Islander: {number_of_asian_pac_islander}")
    number_of_other = len(df[df['race_Other'] == 1])
    print(f"Number of Other: {number_of_other}")
    print()

    print("Percentage of genders that earn more than 50k")
    df_high_income = df[df["income"] == 1]

    percentage_of_male = len(df_high_income[df_high_income['sex_Male'] == 1]) / number_of_male
    print(f"Number of males: {percentage_of_male}")
    percentage_of_female = len(df_high_income[df_high_income['sex_Female'] == 1]) / number_of_female
    print(f"Number of females: {percentage_of_female}")
    print()

    print("Percentage of race that earn more than 50k")
    number_of_white_high = len(df_high_income[df_high_income['race_White'] == 1])
    percentage_of_white = number_of_white_high / number_of_white
    print(f"Number of white: {percentage_of_white}")
    number_of_black_high = len(df_high_income[df_high_income['race_Black'] == 1])
    percentage_of_black = number_of_black_high / number_of_black
    print(f"Number of black: {percentage_of_black}")
    number_of_amer_indian_eskimo_high = len(df_high_income[df_high_income['race_Amer-Indian-Eskimo'] == 1])
    percentage_of_amer_indian_eskimo = number_of_amer_indian_eskimo_high / number_of_amer_indian_eskimo
    print(f"Number of Amer-Indian-Eskimo: {percentage_of_amer_indian_eskimo}")
    number_of_asian_pac_islander_high = len(df_high_income[df_high_income['race_Asian-Pac-Islander'] == 1])
    percentage_of_asian_pac_islander = number_of_asian_pac_islander_high / number_of_asian_pac_islander
    print(f"Number of Asian-Pac-Islander: {percentage_of_asian_pac_islander}")
    number_of_other_high = len(df_high_income[df_high_income['race_Asian-Pac-Islander'] == 1])
    percentage_of_other = number_of_other_high / number_of_other
    print(f"Number of Other: {percentage_of_other}")
    print()