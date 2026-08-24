from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from src.dnn import TorchMLPRegressor


def create_model(
    model_name: str,
    parameters: dict,
    random_state: int = 42,
):
    """Create a regression model from its config name and parameters."""

    parameters = parameters.copy()

    sklearn_models = {
        "linear_regression": LinearRegression,
        "ridge": Ridge,
        "lasso": Lasso,
        "elastic_net": ElasticNet,
        "knn": KNeighborsRegressor,
        "decision_tree": DecisionTreeRegressor,
        "random_forest": RandomForestRegressor,
        "dnn": TorchMLPRegressor,
    }

    if model_name in {"decision_tree", "random_forest", "dnn"}:
        parameters.setdefault("random_state", random_state)

    if model_name in sklearn_models:
        return sklearn_models[model_name](**parameters)

    if model_name == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as error:
            raise ImportError(
                "Install xgboost to enable model 'xgboost'."
            ) from error

        parameters.setdefault("random_state", random_state)
        parameters.setdefault("objective", "reg:squarederror")
        parameters.setdefault("n_jobs", -1)
        return XGBRegressor(**parameters)

    if model_name == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as error:
            raise ImportError(
                "Install lightgbm to enable model 'lightgbm'."
            ) from error

        parameters.setdefault("random_state", random_state)
        parameters.setdefault("verbosity", -1)
        return LGBMRegressor(**parameters)

    if model_name == "catboost":
        try:
            from catboost import CatBoostRegressor
        except ImportError as error:
            raise ImportError(
                "Install catboost to enable model 'catboost'."
            ) from error

        parameters.setdefault("random_seed", random_state)
        parameters.setdefault("verbose", False)
        return CatBoostRegressor(**parameters)

    available_models = sorted(
        [*sklearn_models, "xgboost", "lightgbm", "catboost"]
    )
    raise ValueError(
        f"Unknown model '{model_name}'. Available models: {available_models}"
    )