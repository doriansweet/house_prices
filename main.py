import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import KFold, ParameterGrid, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.data import load_data
from src.features import create_features, prepare_features
from src.models import create_model
from src.preprocessing import create_preprocessor, to_dense


def load_config(path: str = "config.yaml") -> dict:
    """Load project configuration from YAML."""

    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file)


def prepare_datasets(config: dict):
    """Load data, prepare types, and create configured feature groups."""

    train_data, test_data = load_data(
        train_path=config["data"]["train_path"],
        test_path=config["data"]["test_path"],
    )
    target = config["data"]["target"]
    id_column = config["data"]["id_column"]

    X = train_data.drop(columns=[target, id_column])
    y = np.log1p(train_data[target])
    X_test = test_data.drop(columns=[id_column])
    test_ids = test_data[id_column].copy()

    X = prepare_features(X)
    X_test = prepare_features(X_test)

    if config["features"]["enabled"]:
        groups = config["features"]["groups"]
        X = create_features(X, groups=groups)
        X_test = create_features(X_test, groups=groups)

    if not X.columns.equals(X_test.columns):
        raise ValueError("Train and test feature columns do not match.")

    return X, y, X_test, test_ids


def build_pipeline(
    X,
    model_name: str,
    model_parameters: dict,
    model_config: dict,
    config: dict,
) -> Pipeline:
    """Build preprocessing and estimator pipeline for one model candidate."""

    profile_name = model_config["preprocessing"]
    profile = config["preprocessing_profiles"][profile_name]
    preprocessor = create_preprocessor(X, profile)
    model = create_model(
        model_name=model_name,
        parameters=model_parameters,
        random_state=config["project"]["random_state"],
    )

    steps = [("preprocessor", preprocessor)]
    if model_name == "dnn":
        steps.append(
            (
                "to_dense",
                FunctionTransformer(to_dense, accept_sparse=True),
            )
        )
    steps.append(("model", model))
    return Pipeline(steps)


def save_results(rows: list[dict], results_path: str) -> None:
    """Append experiment rows to a CSV results table."""

    path = Path(results_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_results = pd.DataFrame(rows)

    if path.exists():
        previous_results = pd.read_csv(path)
        new_results = pd.concat(
            [previous_results, new_results],
            ignore_index=True,
        )

    new_results.to_csv(path, index=False)


def evaluate_candidate(
    X,
    y,
    model_name: str,
    parameters: dict,
    model_config: dict,
    config: dict,
    cv: KFold,
    run_id: str,
):
    """Evaluate one model and parameter combination on shared CV folds."""

    started_at = time.perf_counter()
    profile_name = model_config["preprocessing"]

    try:
        pipeline = build_pipeline(
            X=X,
            model_name=model_name,
            model_parameters=parameters,
            model_config=model_config,
            config=config,
        )

        n_jobs = 1 if model_name == "dnn" else config["validation"].get("n_jobs", 1)
        scores = -cross_val_score(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=config["validation"]["scoring"],
            n_jobs=n_jobs,
            error_score="raise",
        )

        row = {
            "run_id": run_id,
            "model": model_name,
            "parameters": json.dumps(parameters, sort_keys=True),
            "preprocessing": profile_name,
            "feature_groups": json.dumps(config["features"].get("groups", [])),
            "fold_scores": json.dumps(scores.tolist()),
            "cv_rmse": float(scores.mean()),
            "cv_std": float(scores.std()),
            "duration_seconds": time.perf_counter() - started_at,
            "status": "success",
            "error": "",
        }
        return row, pipeline

    except Exception as error:
        row = {
            "run_id": run_id,
            "model": model_name,
            "parameters": json.dumps(parameters, sort_keys=True),
            "preprocessing": profile_name,
            "feature_groups": json.dumps(config["features"].get("groups", [])),
            "fold_scores": "[]",
            "cv_rmse": np.nan,
            "cv_std": np.nan,
            "duration_seconds": time.perf_counter() - started_at,
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
        }
        return row, None


def rmse(y_true, y_pred) -> float:
    """Calculate root mean squared error."""

    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def save_submission(
    test_ids,
    predictions_log,
    config: dict,
    path: str,
) -> None:
    """Convert log predictions to prices and save a Kaggle submission."""

    target = config["data"]["target"]
    id_column = config["data"]["id_column"]
    predictions = np.expm1(predictions_log).clip(min=0)
    submission = pd.DataFrame(
        {
            id_column: test_ids,
            target: predictions,
        }
    )
    submission_path = Path(path)
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(submission_path, index=False)


def fit_fold_ensemble(
    X,
    y,
    X_test,
    model_spec: dict,
    config: dict,
    cv: KFold,
) -> dict:
    """Train one model on every fold and average its test predictions."""

    oof_predictions = np.zeros(len(X), dtype=float)
    test_predictions = np.zeros(len(X_test), dtype=float)
    fold_scores = []
    n_splits = cv.get_n_splits()

    for fold_number, (train_index, valid_index) in enumerate(
        cv.split(X, y),
        start=1,
    ):
        pipeline = build_pipeline(
            X=X,
            model_name=model_spec["model_name"],
            model_parameters=model_spec["parameters"],
            model_config=model_spec["model_config"],
            config=config,
        )
        X_train = X.iloc[train_index]
        X_valid = X.iloc[valid_index]
        y_train = y.iloc[train_index]
        y_valid = y.iloc[valid_index]

        pipeline.fit(X_train, y_train)
        valid_predictions = pipeline.predict(X_valid)
        fold_score = rmse(y_valid, valid_predictions)
        fold_scores.append(fold_score)
        oof_predictions[valid_index] = valid_predictions
        test_predictions += pipeline.predict(X_test) / n_splits

        print(
            f"{model_spec['model_name']} fold {fold_number}: "
            f"RMSE={fold_score:.5f}"
        )

    return {
        "model_name": model_spec["model_name"],
        "parameters": model_spec["parameters"],
        "preprocessing": model_spec["model_config"]["preprocessing"],
        "oof_predictions": oof_predictions,
        "test_predictions": test_predictions,
        "fold_scores": fold_scores,
        "cv_rmse": float(np.mean(fold_scores)),
        "cv_std": float(np.std(fold_scores)),
        "oof_rmse": rmse(y, oof_predictions),
    }


def build_average_ensemble(
    fold_results: list[dict],
    weights: list[float] | None,
    y,
    cv: KFold,
) -> dict:
    """Average OOF and test predictions from several fold-trained models."""

    if not fold_results:
        raise ValueError("No fold results were provided for the ensemble.")

    if weights is None:
        weights_array = np.ones(len(fold_results), dtype=float)
    else:
        if len(weights) != len(fold_results):
            raise ValueError("Ensemble weights must match the number of models.")
        weights_array = np.asarray(weights, dtype=float)

    weights_array = weights_array / weights_array.sum()
    oof_predictions = np.zeros(len(y), dtype=float)
    test_predictions = np.zeros_like(
        fold_results[0]["test_predictions"],
        dtype=float,
    )

    for weight, result in zip(weights_array, fold_results):
        oof_predictions += weight * result["oof_predictions"]
        test_predictions += weight * result["test_predictions"]

    fold_scores = []
    for _, valid_index in cv.split(np.zeros(len(y)), y):
        fold_scores.append(rmse(y.iloc[valid_index], oof_predictions[valid_index]))

    return {
        "model_name": "average_ensemble",
        "members": [result["model_name"] for result in fold_results],
        "weights": weights_array.tolist(),
        "oof_predictions": oof_predictions,
        "test_predictions": test_predictions,
        "fold_scores": fold_scores,
        "cv_rmse": float(np.mean(fold_scores)),
        "cv_std": float(np.std(fold_scores)),
        "oof_rmse": rmse(y, oof_predictions),
    }


def main():
    """Evaluate all enabled models and create a submission from the best one."""

    config = load_config()
    X, y, X_test, test_ids = prepare_datasets(config)

    cv = KFold(
        n_splits=config["validation"]["n_splits"],
        shuffle=config["validation"]["shuffle"],
        random_state=config["project"]["random_state"],
    )
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rows = []
    best_spec = None
    best_by_model = {}

    for model_name, model_config in config["models"].items():
        if not model_config.get("enabled", False):
            continue

        parameter_grid = model_config.get("param_grid", {})
        for parameters in ParameterGrid(parameter_grid):
            print(f"\nModel: {model_name}; parameters: {parameters}")
            row, _ = evaluate_candidate(
                X=X,
                y=y,
                model_name=model_name,
                parameters=parameters,
                model_config=model_config,
                config=config,
                cv=cv,
                run_id=run_id,
            )
            rows.append(row)

            if row["status"] == "success":
                print(f"CV RMSE: {row['cv_rmse']:.5f}")
                print(f"CV std:  {row['cv_std']:.5f}")
                if best_spec is None or row["cv_rmse"] < best_spec["cv_rmse"]:
                    best_spec = {
                        "model_name": model_name,
                        "parameters": parameters,
                        "model_config": model_config,
                        "cv_rmse": row["cv_rmse"],
                    }
                current_model_best = best_by_model.get(model_name)
                if (
                    current_model_best is None
                    or row["cv_rmse"] < current_model_best["cv_rmse"]
                ):
                    best_by_model[model_name] = {
                        "model_name": model_name,
                        "parameters": parameters,
                        "model_config": model_config,
                        "cv_rmse": row["cv_rmse"],
                    }
            else:
                print(f"Skipped: {row['error']}")

    save_results(rows, config["output"]["results_path"])

    if best_spec is None:
        raise RuntimeError("No model completed successfully. Check experiments.csv.")

    print("\nBest model:", best_spec["model_name"])
    print("Best parameters:", best_spec["parameters"])
    print(f"Best CV RMSE: {best_spec['cv_rmse']:.5f}")

    best_pipeline = build_pipeline(
        X=X,
        model_name=best_spec["model_name"],
        model_parameters=best_spec["parameters"],
        model_config=best_spec["model_config"],
        config=config,
    )
    best_pipeline.fit(X, y)

    predictions_log = best_pipeline.predict(X_test)
    submission_path = Path(config["output"]["submission_path"])
    save_submission(
        test_ids=test_ids,
        predictions_log=predictions_log,
        config=config,
        path=str(submission_path),
    )

    model_path = Path(config["output"]["best_model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, model_path)

    ensemble_config = config.get("ensemble", {})
    if ensemble_config.get("enabled", False):
        requested_models = ensemble_config["models"]
        missing_models = [
            model_name
            for model_name in requested_models
            if model_name not in best_by_model
        ]
        if missing_models:
            raise ValueError(
                "Cannot build ensemble because these models did not finish: "
                f"{missing_models}"
            )

        print("\nTraining 5-fold models for averaging...")
        fold_results = []
        ensemble_rows = []
        fold_directory = Path(ensemble_config["submission_directory"])
        fold_directory.mkdir(parents=True, exist_ok=True)

        for model_name in requested_models:
            started_at = time.perf_counter()
            result = fit_fold_ensemble(
                X=X,
                y=y,
                X_test=X_test,
                model_spec=best_by_model[model_name],
                config=config,
                cv=cv,
            )
            fold_results.append(result)
            model_submission_path = fold_directory / f"{model_name}_5fold.csv"
            save_submission(
                test_ids=test_ids,
                predictions_log=result["test_predictions"],
                config=config,
                path=str(model_submission_path),
            )
            ensemble_rows.append(
                {
                    "run_id": run_id,
                    "model": f"{model_name}_5fold",
                    "parameters": json.dumps(result["parameters"], sort_keys=True),
                    "preprocessing": result["preprocessing"],
                    "feature_groups": json.dumps(config["features"].get("groups", [])),
                    "fold_scores": json.dumps(result["fold_scores"]),
                    "cv_rmse": result["cv_rmse"],
                    "cv_std": result["cv_std"],
                    "duration_seconds": time.perf_counter() - started_at,
                    "status": "success",
                    "error": "",
                }
            )
            print(
                f"{model_name} 5-fold mean RMSE: {result['cv_rmse']:.5f}; "
                f"OOF RMSE: {result['oof_rmse']:.5f}"
            )

        average_result = build_average_ensemble(
            fold_results=fold_results,
            weights=ensemble_config.get("weights"),
            y=y,
            cv=cv,
        )
        average_submission_path = fold_directory / "average_ensemble_5fold.csv"
        save_submission(
            test_ids=test_ids,
            predictions_log=average_result["test_predictions"],
            config=config,
            path=str(average_submission_path),
        )
        ensemble_rows.append(
            {
                "run_id": run_id,
                "model": "average_ensemble_5fold",
                "parameters": json.dumps(
                    {
                        "members": average_result["members"],
                        "weights": average_result["weights"],
                    },
                    sort_keys=True,
                ),
                "preprocessing": "mixed",
                "feature_groups": json.dumps(config["features"].get("groups", [])),
                "fold_scores": json.dumps(average_result["fold_scores"]),
                "cv_rmse": average_result["cv_rmse"],
                "cv_std": average_result["cv_std"],
                "duration_seconds": 0.0,
                "status": "success",
                "error": "",
            }
        )
        save_results(ensemble_rows, config["output"]["results_path"])
        print(
            "Average ensemble 5-fold mean RMSE: "
            f"{average_result['cv_rmse']:.5f}; "
            f"OOF RMSE: {average_result['oof_rmse']:.5f}"
        )
        print(f"5-fold submissions saved in: {fold_directory}")

    print(f"Submission saved: {submission_path}")
    print(f"Best model saved: {model_path}")
    print(f"Results saved: {config['output']['results_path']}")


if __name__ == "__main__":
    main()