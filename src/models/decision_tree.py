import pandas as pd
from sklearn.tree import DecisionTreeClassifier, _tree
from src.models import BaseModel
import matplotlib.pyplot as plt
import seaborn as sns

class DecisionTree:
    _surrogate_model: DecisionTreeClassifier


    def __init__(self, params: dict):
        self._surrogate_model = DecisionTreeClassifier(**params)

    def fit(self, X_train, y_train):
        assert len(y_train.shape) == 1, "y_train should be 1D"

        self._surrogate_model.fit(X_train, y_train)

    def predict(self, X):
        return self._surrogate_model.predict(X)

    def feature_importance(self):
        return self._surrogate_model.feature_importances_

    def plot_feature_importance(self, feature_names: list[str]):
        df = pd.DataFrame(columns=["feature", "importance"], data=zip(feature_names, self.feature_importance()))
        df = df.sort_values("importance", ascending=True)
        df = df[df["importance"] > 0]

        df.plot(kind='barh', x='feature', y='importance')
        plt.show()

    def print_decision_paths(self, feature_names):
        tree_ = self._surrogate_model.tree_
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]

        def recurse(node, depth, parent_rule):
            indent = "  " * depth
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]
                rule_left = f"{name} <= {threshold:.2f}"
                rule_right = f"{name} > {threshold:.2f}"
                if depth > 0:
                    rule_left = f"{parent_rule} AND {rule_left}"
                    rule_right = f"{parent_rule} AND {rule_right}"
                recurse(tree_.children_left[node], depth + 1, rule_left)
                recurse(tree_.children_right[node], depth + 1, rule_right)
            else:
                predictions = tree_.value[node].flatten()
                sample_counts = tree_.n_node_samples[node]
                if sample_counts < 500:
                    return
                print(
                    f"{indent}Leaf({node}): {parent_rule} -> Predict = {[round(val, 2) for val in predictions]}, Samples per class = {predictions}, Total samples = {sample_counts}")

        # Start the recursion from the root node
        recurse(0, 0, "IF")


