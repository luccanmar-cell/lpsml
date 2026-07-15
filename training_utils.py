from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

try:
    import optuna
except ImportError:
    optuna = None


MODEL_TYPES: dict[str, type[BaseEstimator]] = {
    "elastic_net": ElasticNet,
    "extra_trees": ExtraTreesRegressor,
    "gradient_boosting": GradientBoostingRegressor,
    "hist_gradient_boosting": HistGradientBoostingRegressor,
    "random_forest": RandomForestRegressor,
    "ridge": Ridge,
}

SCALER_TYPES: dict[str, type[BaseEstimator] | None] = {
    "none": None,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
    "standard": StandardScaler,
}


def load_json_config(config_path: str | Path) -> dict[str, Any]:
    """Load model, split, and cross-validation settings from JSON."""
    with Path(config_path).open("r", encoding="utf-8") as config_file:
        return json.load(config_file)


def load_training_data(
    dataset_path: str | Path,
    target_columns: list[str],
    total_target_column: str,
    drop_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load numeric features and component Prima targets from parquet."""
    df = pd.read_parquet(dataset_path).apply(pd.to_numeric, errors="coerce")
    required_targets = [*target_columns, total_target_column]
    missing_targets = [column for column in required_targets if column not in df.columns]
    if missing_targets:
        raise ValueError(f"Target columns were not found: {missing_targets}")

    df = df.dropna(subset=required_targets)
    excluded_columns = [*required_targets, *(drop_columns or [])]
    features = df.drop(columns=excluded_columns, errors="ignore")
    targets = df[target_columns]

    if features.empty:
        raise ValueError("No feature columns remain after dropping excluded columns.")
    if features.isna().any().any():
        missing_columns = features.columns[features.isna().any()].tolist()
        raise ValueError(f"Feature columns contain missing values: {missing_columns}")

    return features, targets


def split_train_test(
    features: pd.DataFrame,
    target: pd.DataFrame,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reserve the test set before any cross-validation or model selection."""
    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
    )


def build_pipeline(
    model_type: str,
    scaler_type: str,
    fixed_params: dict[str, Any] | None = None,
) -> Pipeline:
    """Build a model pipeline from names supported by the JSON configuration."""
    if model_type not in MODEL_TYPES:
        supported = ", ".join(sorted(MODEL_TYPES))
        raise ValueError(f"Unsupported model type '{model_type}'. Choose from: {supported}.")
    if scaler_type not in SCALER_TYPES:
        supported = ", ".join(sorted(SCALER_TYPES))
        raise ValueError(f"Unsupported scaler type '{scaler_type}'. Choose from: {supported}.")

    base_model = MODEL_TYPES[model_type](**(fixed_params or {}))
    model = MultiOutputRegressor(base_model, n_jobs=1)
    scaler_class = SCALER_TYPES[scaler_type]
    scaler: BaseEstimator | str = scaler_class() if scaler_class else "passthrough"
    return Pipeline([("scaler", scaler), ("model", model)])


def build_parameter_grid(
    parameter_grid: dict[str, list[Any]] | None,
) -> dict[str, list[Any]]:
    """Prefix model parameters for GridSearchCV's Cartesian-product grid."""
    grid = parameter_grid or {}
    invalid_parameters = [name for name, values in grid.items() if not isinstance(values, list)]
    if invalid_parameters:
        raise ValueError(
            "Every parameter grid value must be a JSON list. Invalid parameters: "
            + ", ".join(invalid_parameters)
        )
    return {f"model__estimator__{name}": values for name, values in grid.items()}


def balanced_multioutput_mse(
    y_true: Any,
    y_pred: Any,
    final_weight: float = 0.5,
) -> float:
    """Balance normalized component MSE with MSE of their derived total."""
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    component_mse = np.mean((actual - predicted) ** 2, axis=0)
    component_scale = np.maximum(np.var(actual, axis=0), 1.0)
    normalized_component_mse = float(np.mean(component_mse / component_scale))

    actual_total = actual.sum(axis=1)
    predicted_total = predicted.sum(axis=1)
    total_mse = float(np.mean((actual_total - predicted_total) ** 2))
    total_scale = max(float(np.var(actual_total)), 1.0)
    normalized_total_mse = total_mse / total_scale
    return (
        (1.0 - final_weight) * normalized_component_mse
        + final_weight * normalized_total_mse
    )


def make_balanced_mse_scorer(final_weight: float) -> Any:
    if not 0 <= final_weight <= 1:
        raise ValueError("final_weight must be between 0 and 1.")
    return make_scorer(
        balanced_multioutput_mse,
        greater_is_better=False,
        final_weight=final_weight,
    )


def suggest_optuna_parameters(
    trial: Any,
    search_space: dict[str, Any],
) -> dict[str, Any]:
    """Generate one model parameter set from an Optuna search space."""
    parameters: dict[str, Any] = {}
    for name, spec in search_space.items():
        if isinstance(spec, list):
            parameters[name] = trial.suggest_categorical(name, spec)
        elif spec["type"] == "int":
            parameters[name] = trial.suggest_int(
                name,
                int(spec["low"]),
                int(spec["high"]),
                step=int(spec.get("step", 1)),
                log=bool(spec.get("log", False)),
            )
        elif spec["type"] == "float":
            parameters[name] = trial.suggest_float(
                name,
                float(spec["low"]),
                float(spec["high"]),
                step=spec.get("step"),
                log=bool(spec.get("log", False)),
            )
        elif spec["type"] == "categorical":
            parameters[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"Unsupported Optuna parameter type for '{name}'.")
    return parameters


def run_grid_search(
    pipeline: Pipeline,
    model_config: dict[str, Any],
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    cv: KFold,
    n_jobs: int,
    scorer: Any,
) -> tuple[Pipeline, dict[str, Any], float]:
    """Exhaustively select parameters using balanced component/final MSE."""
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=build_parameter_grid(model_config.get("param_grid")),
        scoring=scorer,
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
    )
    search.fit(x_train, y_train)
    best_params = {
        name.removeprefix("model__estimator__"): value
        for name, value in search.best_params_.items()
    }
    return search.best_estimator_, best_params, -float(search.best_score_)


def run_optuna_search(
    pipeline: Pipeline,
    model_config: dict[str, Any],
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    cv: KFold,
    n_jobs: int,
    search_config: dict[str, Any],
    random_state: int,
    scorer: Any,
) -> tuple[Pipeline, dict[str, Any], float]:
    """Use Optuna to minimize cross-validated MSE."""
    if optuna is None:
        raise RuntimeError("Optuna search requires the optional 'optuna' package.")

    search_space = model_config.get("optuna_space", model_config.get("param_grid", {}))
    trial_n_jobs = int(search_config.get("trial_n_jobs", 1))
    trial_cv_n_jobs = n_jobs
    pipeline_params = pipeline.get_params()
    model_n_jobs = {
        name: value
        for name, value in pipeline_params.items()
        if name.startswith("model__") and name.endswith("n_jobs")
    }
    if trial_n_jobs != 1:
        # Parallelize at one level only to avoid multiplying trial, CV, and model workers.
        trial_cv_n_jobs = 1
        pipeline.set_params(**{name: 1 for name in model_n_jobs})

    def objective(trial: Any) -> float:
        parameters = suggest_optuna_parameters(trial, search_space)
        candidate = clone(pipeline).set_params(
            **{f"model__estimator__{name}": value for name, value in parameters.items()}
        )
        scores = cross_val_score(
            candidate,
            x_train,
            y_train,
            scoring=scorer,
            cv=cv,
            n_jobs=trial_cv_n_jobs,
        )
        return -float(scores.mean())

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(
        objective,
        n_trials=int(search_config.get("n_trials", 100)),
        timeout=search_config.get("timeout_seconds"),
        show_progress_bar=bool(search_config.get("show_progress_bar", False)),
        n_jobs=trial_n_jobs,
    )

    best_params = dict(study.best_trial.params)
    best_pipeline = clone(pipeline).set_params(
        **{f"model__estimator__{name}": value for name, value in best_params.items()}
    )
    best_pipeline.set_params(**model_n_jobs)
    best_pipeline.fit(x_train, y_train)
    return best_pipeline, best_params, float(study.best_value)


def search_models(
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[Pipeline, dict[str, Any], list[dict[str, Any]]]:
    """Select a model by cross-validated training MSE using grid or Optuna."""
    cv_config = config.get("cv", {})
    search_config = config.get("search", {})
    search_method = search_config.get("method", "grid")
    if search_method not in {"grid", "optuna"}:
        raise ValueError("search.method must be either 'grid' or 'optuna'.")

    random_state = int(config.get("random_state", 42))
    shuffle = bool(cv_config.get("shuffle", True))
    cv = KFold(
        n_splits=int(cv_config.get("n_splits", 5)),
        shuffle=shuffle,
        random_state=random_state if shuffle else None,
    )
    n_jobs = int(cv_config.get("n_jobs", -1))
    final_weight = float(config.get("final_weight", 0.5))
    scorer = make_balanced_mse_scorer(final_weight)

    results: list[dict[str, Any]] = []
    for model_config in config.get("models", []):
        if not model_config.get("enabled", True):
            continue

        model_name = model_config.get("name", model_config["type"])
        for scaler_type in model_config.get("scalers", ["none"]):
            pipeline = build_pipeline(
                model_config["type"],
                scaler_type,
                fixed_params=model_config.get("fixed_params"),
            )
            if search_method == "grid":
                estimator, best_params, cv_mse = run_grid_search(
                    pipeline, model_config, x_train, y_train, cv, n_jobs, scorer
                )
            else:
                estimator, best_params, cv_mse = run_optuna_search(
                    pipeline,
                    model_config,
                    x_train,
                    y_train,
                    cv,
                    n_jobs,
                    search_config,
                    random_state,
                    scorer,
                )

            results.append(
                {
                    "model": model_name,
                    "model_type": model_config["type"],
                    "scaler": scaler_type,
                    "search_method": search_method,
                    "cv_balanced_mse": cv_mse,
                    "best_params": best_params,
                    "estimator": estimator,
                }
            )

    if not results:
        raise ValueError("The configuration does not contain any enabled models.")

    best_result = min(results, key=lambda result: result["cv_balanced_mse"])
    return best_result["estimator"], best_result, results
