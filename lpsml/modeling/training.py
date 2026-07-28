from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

from joblib import Parallel, delayed, effective_n_jobs, parallel_backend
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import ParameterGrid, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from .torch_regressor import NonnegativeJointRegressor

try:
    import optuna
except ImportError:
    optuna = None


MODEL_TYPES: dict[str, type[BaseEstimator]] = {
    "extra_trees": ExtraTreesRegressor,
    "pytorch": NonnegativeJointRegressor,
    "random_forest": RandomForestRegressor,
}

JOINT_LOSS_MODEL_TYPES = {"pytorch"}
MODEL_PARAMETER_PREFIX = "model__"

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
    if df[required_targets].lt(0).any().any():
        negative_columns = df[required_targets].columns[
            df[required_targets].lt(0).any()
        ].tolist()
        raise ValueError(
            "Premium targets must be nonnegative. Rebuild the dataset; "
            f"negative values remain in: {negative_columns}"
        )
    excluded_columns = [*required_targets, *(drop_columns or [])]
    features = df.drop(columns=excluded_columns, errors="ignore")
    targets = df[target_columns]

    if features.empty:
        raise ValueError("No feature columns remain after dropping excluded columns.")
    if features.isna().any().any():
        missing_columns = features.columns[features.isna().any()].tolist()
        raise ValueError(f"Feature columns contain missing values: {missing_columns}")

    return features, targets


def build_joint_strata(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Build stable labels for stratifying on a combination of columns."""
    missing_columns = [column for column in columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Stratification columns were not found: {missing_columns}")
    if not columns:
        raise ValueError("At least one stratification column is required.")
    return (
        frame[columns]
        .astype("string")
        .fillna("Missing")
        .agg("\x1f".join, axis=1)
        .rename("Stratum")
    )


def split_train_test(
    features: pd.DataFrame,
    target: pd.DataFrame,
    strata: pd.Series,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reserve a holdout set while preserving every supplied stratum."""
    if not features.index.equals(target.index) or not features.index.equals(strata.index):
        raise ValueError("Features, targets, and strata must have identical indices.")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    stratum_counts = strata.value_counts()
    if stratum_counts.empty or stratum_counts.min() < 2:
        raise ValueError(
            "Every stratification pair needs at least two rows. "
            "Increase the data-processing pair threshold or rebuild the dataset."
        )
    test_rows = int(np.ceil(len(features) * test_size))
    train_rows = len(features) - test_rows
    if len(stratum_counts) > min(train_rows, test_rows):
        raise ValueError(
            "The requested split is too small to place every stratification pair "
            "in both train and test."
        )

    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=strata,
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

    model = MODEL_TYPES[model_type](**(fixed_params or {}))
    scaler_class = SCALER_TYPES[scaler_type]
    scaler: BaseEstimator | str = scaler_class() if scaler_class else "passthrough"
    return Pipeline([("scaler", scaler), ("model", model)])


def predict_premium_components(model: Any, features: pd.DataFrame) -> np.ndarray:
    """Predict premium components and enforce the nonnegative model contract."""
    predictions = np.asarray(model.predict(features), dtype=float)
    if not np.isfinite(predictions).all():
        raise ValueError("The model produced a non-finite premium component.")
    if np.any(predictions < 0):
        minimum = float(predictions.min())
        raise ValueError(
            "The model produced a negative premium component "
            f"(minimum={minimum:.6f})."
        )
    return predictions


def build_parameter_grid(
    parameter_grid: dict[str, list[Any]] | None,
) -> dict[str, list[Any]]:
    """Prefix model parameters for the Cartesian-product grid search."""
    grid = parameter_grid or {}
    invalid_parameters = [name for name, values in grid.items() if not isinstance(values, list)]
    if invalid_parameters:
        raise ValueError(
            "Every parameter grid value must be a JSON list. Invalid parameters: "
            + ", ".join(invalid_parameters)
        )
    return {f"{MODEL_PARAMETER_PREFIX}{name}": values for name, values in grid.items()}


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


def worst_group_mape(
    y_true: Any,
    y_pred: Any,
    groups: pd.Series,
) -> float:
    """Return the largest final-premium MAPE among predefined row groups."""
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    if actual.shape != predicted.shape or actual.ndim != 2:
        raise ValueError("Actual and predicted component targets must be matching matrices.")
    if len(groups) != len(actual):
        raise ValueError("Group labels must have one value per target row.")

    actual_total = actual.sum(axis=1)
    predicted_total = predicted.sum(axis=1)
    denominator = np.abs(actual_total)
    if np.any(denominator <= np.finfo(float).eps):
        raise ValueError(
            "Worst-group MAPE requires a strictly positive final premium for every row."
        )
    percentage_error = np.abs(actual_total - predicted_total) / denominator * 100
    grouped_mape = (
        pd.DataFrame(
            {
                "group": pd.Series(
                    np.asarray(groups, dtype=object),
                    dtype="string",
                ).fillna("Missing"),
                "percentage_error": percentage_error,
            }
        )
        .groupby("group", sort=True)["percentage_error"]
        .mean()
    )
    if grouped_mape.empty:
        raise ValueError("Worst-group MAPE requires at least one group.")
    return float(grouped_mape.max())


def cross_validated_worst_group_mape(
    pipeline: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    strata: pd.Series,
    cv: StratifiedKFold,
    n_jobs: int,
) -> float:
    """Score complete out-of-fold predictions by their worst pair-level MAPE."""
    if (
        not x_train.index.equals(y_train.index)
        or not x_train.index.equals(strata.index)
    ):
        raise ValueError("Features, targets, and strata must have identical indices.")

    tuning_pipeline, inner_threads = configure_tuning_parallelism(pipeline, n_jobs)
    splits = list(cv.split(x_train, strata))

    def fit_and_predict(
        training_positions: np.ndarray,
        validation_positions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        estimator = clone(tuning_pipeline)
        estimator.fit(
            x_train.iloc[training_positions],
            y_train.iloc[training_positions],
        )
        predictions = predict_premium_components(
            estimator,
            x_train.iloc[validation_positions],
        )
        return validation_positions, predictions

    with parallel_backend("loky", inner_max_num_threads=inner_threads):
        fold_predictions = Parallel(
            n_jobs=n_jobs,
            pre_dispatch=n_jobs if n_jobs > 0 else "n_jobs",
        )(
            delayed(fit_and_predict)(training_positions, validation_positions)
            for training_positions, validation_positions in splits
        )

    out_of_fold_predictions = np.full(y_train.shape, np.nan, dtype=float)
    for validation_positions, predictions in fold_predictions:
        out_of_fold_predictions[validation_positions] = predictions
    if not np.isfinite(out_of_fold_predictions).all():
        raise RuntimeError("Cross-validation did not predict every training row.")

    return worst_group_mape(
        y_train,
        out_of_fold_predictions,
        strata,
    )


def configure_tuning_parallelism(
    pipeline: Pipeline,
    outer_n_jobs: int,
) -> tuple[Pipeline, int]:
    """Bound nested estimator threads so concurrent CV fits share CPUs predictably."""
    if outer_n_jobs == 0:
        raise ValueError("cv.n_jobs cannot be 0.")

    outer_workers = effective_n_jobs(outer_n_jobs)
    inner_threads = max(1, (os.cpu_count() or 1) // outer_workers)
    parallel_params: dict[str, int] = {}
    for name in pipeline.get_params():
        if not (name.startswith("model__") and name.endswith("n_jobs")):
            continue
        parallel_params[name] = inner_threads

    configured = clone(pipeline)
    if parallel_params:
        configured.set_params(**parallel_params)
    return configured, inner_threads


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
    strata: pd.Series,
    cv: StratifiedKFold,
    n_jobs: int,
) -> tuple[Pipeline, dict[str, Any], float]:
    """Exhaustively select parameters by out-of-fold worst-pair MAPE."""
    scores: list[tuple[float, dict[str, Any]]] = []
    for prefixed_params in ParameterGrid(
        build_parameter_grid(model_config.get("param_grid"))
    ):
        candidate = clone(pipeline).set_params(**prefixed_params)
        score = cross_validated_worst_group_mape(
            candidate,
            x_train,
            y_train,
            strata,
            cv,
            n_jobs,
        )
        scores.append((score, prefixed_params))

    best_score, best_prefixed_params = min(scores, key=lambda result: result[0])
    best_params = {
        name.removeprefix(MODEL_PARAMETER_PREFIX): value
        for name, value in best_prefixed_params.items()
    }
    best_pipeline = clone(pipeline).set_params(
        **{f"{MODEL_PARAMETER_PREFIX}{name}": value for name, value in best_params.items()}
    )
    best_pipeline.fit(x_train, y_train)
    return best_pipeline, best_params, best_score


def run_optuna_search(
    pipeline: Pipeline,
    model_config: dict[str, Any],
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    strata: pd.Series,
    cv: StratifiedKFold,
    n_jobs: int,
    search_config: dict[str, Any],
    random_state: int,
) -> tuple[Pipeline, dict[str, Any], float]:
    """Use Optuna to minimize out-of-fold worst-pair final-Prima MAPE."""
    if optuna is None:
        raise RuntimeError("Optuna search requires the optional 'optuna' package.")

    search_space = model_config.get("optuna_space", model_config.get("param_grid", {}))
    trial_n_jobs = int(search_config.get("trial_n_jobs", 1))
    if trial_n_jobs != 1:
        raise ValueError(
            "Parallel Optuna trials are disabled to prevent trials, CV folds, and "
            "native estimator threads from competing. Keep search.trial_n_jobs at 1 "
            "and use cv.n_jobs for parallel tuning."
        )
    trial_cv_n_jobs = n_jobs
    pipeline_params = pipeline.get_params()
    model_n_jobs = {
        name: value
        for name, value in pipeline_params.items()
        if name.startswith("model__") and name.endswith("n_jobs")
    }

    def objective(trial: Any) -> float:
        parameters = suggest_optuna_parameters(trial, search_space)
        candidate = clone(pipeline).set_params(
            **{
                f"{MODEL_PARAMETER_PREFIX}{name}": value
                for name, value in parameters.items()
            }
        )
        return cross_validated_worst_group_mape(
            candidate,
            x_train,
            y_train,
            strata,
            cv,
            trial_cv_n_jobs,
        )

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(
        objective,
        n_trials=int(model_config.get("n_trials", search_config.get("n_trials", 100))),
        timeout=model_config.get(
            "timeout_seconds",
            search_config.get("timeout_seconds"),
        ),
        show_progress_bar=bool(search_config.get("show_progress_bar", False)),
        n_jobs=trial_n_jobs,
    )

    best_params = dict(study.best_trial.params)
    best_pipeline = clone(pipeline).set_params(
        **{f"{MODEL_PARAMETER_PREFIX}{name}": value for name, value in best_params.items()}
    )
    best_pipeline.set_params(**model_n_jobs)
    best_pipeline.fit(x_train, y_train)
    return best_pipeline, best_params, float(study.best_value)


def search_models(
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    strata: pd.Series,
    config: dict[str, Any],
) -> tuple[Pipeline, dict[str, Any], list[dict[str, Any]]]:
    """Select a model by pair-stratified out-of-fold worst-pair MAPE."""
    cv_config = config.get("cv", {})
    search_config = config.get("search", {})
    search_method = search_config.get("method", "grid")
    if search_method not in {"grid", "optuna"}:
        raise ValueError("search.method must be either 'grid' or 'optuna'.")
    if (
        not x_train.index.equals(y_train.index)
        or not x_train.index.equals(strata.index)
    ):
        raise ValueError("Features, targets, and strata must have identical indices.")

    random_state = int(config.get("random_state", 42))
    shuffle = bool(cv_config.get("shuffle", True))
    n_splits = int(cv_config.get("n_splits", 5))
    stratum_counts = strata.value_counts()
    if stratum_counts.empty or stratum_counts.min() < n_splits:
        raise ValueError(
            "Every tariff-coverage pair needs at least cv.n_splits training rows."
        )
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state if shuffle else None,
    )
    n_jobs = int(cv_config.get("n_jobs", -1))
    final_weight = float(config.get("final_weight", 0.5))

    results: list[dict[str, Any]] = []
    for model_config in config.get("models", []):
        if not model_config.get("enabled", True):
            continue

        model_name = model_config.get("name", model_config["type"])
        for scaler_type in model_config.get("scalers", ["none"]):
            fixed_params = dict(model_config.get("fixed_params", {}))
            if model_config["type"] in JOINT_LOSS_MODEL_TYPES:
                fixed_params.setdefault("final_weight", final_weight)
                fixed_params.setdefault("random_state", random_state)
            pipeline = build_pipeline(
                model_config["type"],
                scaler_type,
                fixed_params=fixed_params,
            )
            if search_method == "grid":
                estimator, best_params, cv_worst_pair_mape = run_grid_search(
                    pipeline,
                    model_config,
                    x_train,
                    y_train,
                    strata,
                    cv,
                    n_jobs,
                )
            else:
                estimator, best_params, cv_worst_pair_mape = run_optuna_search(
                    pipeline,
                    model_config,
                    x_train,
                    y_train,
                    strata,
                    cv,
                    n_jobs,
                    search_config,
                    random_state,
                )

            results.append(
                {
                    "model": model_name,
                    "model_type": model_config["type"],
                    "multioutput_strategy": (
                        "joint-loss"
                        if model_config["type"] in JOINT_LOSS_MODEL_TYPES
                        else "native"
                    ),
                    "scaler": scaler_type,
                    "search_method": search_method,
                    "search_trials": (
                        int(
                            model_config.get(
                                "n_trials",
                                search_config.get("n_trials", 100),
                            )
                        )
                        if search_method == "optuna"
                        else None
                    ),
                    "cv_worst_pair_mape_percent": cv_worst_pair_mape,
                    "best_params": best_params,
                    "estimator": estimator,
                }
            )

    if not results:
        raise ValueError("The configuration does not contain any enabled models.")

    best_result = min(
        results,
        key=lambda result: result["cv_worst_pair_mape_percent"],
    )
    return best_result["estimator"], best_result, results
