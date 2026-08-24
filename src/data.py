import pandas as pd


def load_data(train_path: str, test_path: str):
    """Загружает обучающую и тестовую выборки."""

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    return train_data, test_data