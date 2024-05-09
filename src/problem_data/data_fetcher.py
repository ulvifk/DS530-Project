import os.path

from ucimlrepo import fetch_ucirepo
import pandas as pd
import pickle


class DataFetcher:

    def get_data(self, id: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        folder_name = "datasets"
        saved_data_path_X = f"{folder_name}/{id}_X.pickle"
        saved_data_path_y = f"{folder_name}/{id}_y.pickle"
        try:
            with open(saved_data_path_X, "rb") as f:
                X = pickle.load(f)
            with open(saved_data_path_y, "rb") as f:
                y = pickle.load(f)

                try:
                    y["income"] = y["income"].apply(lambda x: x.replace(".", ""))
                except:
                    pass
        except FileNotFoundError:
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)

            data_set = fetch_ucirepo(id=id)
            X = data_set.data.features
            y = data_set.data.targets

            try:
                y["income"] = y["income"].apply(lambda x: x.replace(".", ""))
            except:
                pass
            with open(saved_data_path_X, "wb") as f:
                pickle.dump(X, f)
            with open(saved_data_path_y, "wb") as f:
                pickle.dump(y, f)

        return X, y
