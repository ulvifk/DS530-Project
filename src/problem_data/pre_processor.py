import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


class PreProcessor:
    df: pd.DataFrame

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.df.dropna(inplace=True)

        self.cols_to_one_hot = ["workclass", "education", "marital-status", "occupation",
                                "relationship", "native-country"]

        self.cols_to_label_encode = ["income"]

        self._one_hot_encode()
        self._label_encode()

    def _one_hot_encode(self):
        for col in self.cols_to_one_hot:
            try:
                onehot_encoder = OneHotEncoder()
                col_to_encoded = self.df[col].values.reshape(-1, 1)
                self.df.drop(col, axis=1, inplace=True)

                onehot_encoded = onehot_encoder.fit_transform(col_to_encoded).toarray()
                categories = onehot_encoder.categories_[0]
                col_names = [str(col) + "_" + category for category in categories]

                self.df = pd.concat([self.df, pd.DataFrame(onehot_encoded, columns=col_names, index=self.df.index)], axis=1)
            except:
                pass

    def _label_encode(self):
        self.df["income"] = self.df["income"].apply(lambda x: 1 if x == ">50K" else 0)

        try:
            self.df["race"] = self.df["race"].apply(lambda x: 1 if x == "White" else 0)
        except:
            pass
        try:
            self.df["sex"] = self.df["sex"].apply(lambda x: 1 if x == "Male" else 0)
        except:
            pass