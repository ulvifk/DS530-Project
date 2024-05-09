import yaml
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class UserSettings:
    is_smote: bool = False
    scaler: type = MinMaxScaler


def load_settings():
    with open("user_settings.yaml", "r") as f:
        settings = yaml.safe_load(f)["UserSettings"]
        UserSettings.is_smote = settings["is_smote"]
        UserSettings.scaler = StandardScaler if settings["scaler"] == "StandardScaler" else MinMaxScaler

load_settings()