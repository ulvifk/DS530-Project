import pandas as pd
from src.problem_data.data_fetcher import DataFetcher
from src.problem_data.pre_processor import PreProcessor


class ProblemData:
    original_df: pd.DataFrame
    df_X: pd.DataFrame
    df_y: pd.DataFrame
    df: pd.DataFrame
    target_col: str = "income"

    def __init__(self, exclude_cols: list[str] = []):
        data_fetcher = DataFetcher()
        self.df_X, self.df_y = data_fetcher.get_data(2)

        self.df = pd.concat([self.df_X, self.df_y], axis=1)
        self.original_df = self.df.copy()

        self.df = self.df.drop(exclude_cols, axis=1)
        pre_processor = PreProcessor(self.df)
        self.df = pre_processor.df
        self.df_X = self.df[[col for col in self.df.columns if col != self.target_col]]
        self.df_y = self.df[[self.target_col]]





