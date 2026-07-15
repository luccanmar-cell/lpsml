from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def absolute_percentage_errors(
    y_true: Iterable[float],
    y_pred: Iterable[float],
) -> np.ndarray:
    """Return percentage errors, with zero-target rows marked undefined."""
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    denominator = np.abs(actual)
    return np.divide(
        np.abs(actual - predicted),
        denominator,
        out=np.full(actual.shape, np.nan, dtype=float),
        where=denominator > np.finfo(float).eps,
    ) * 100


def compute_regression_metrics(
    y_true: Iterable[float],
    y_pred: Iterable[float],
) -> dict[str, Any]:
    """Compute test MSE, MAE, and MAPE percentage."""
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    percentage_errors = absolute_percentage_errors(actual, predicted)
    valid_percentage_rows = int(np.isfinite(percentage_errors).sum())
    return {
        "mse": float(mean_squared_error(actual, predicted)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "mape_percent": (
            float(np.nanmean(percentage_errors)) if valid_percentage_rows else None
        ),
        "mape_valid_rows": valid_percentage_rows,
        "zero_target_rows": int(len(actual) - valid_percentage_rows),
    }


def compute_multioutput_metrics(
    y_true_components: pd.DataFrame,
    y_pred_components: np.ndarray,
    y_true_total: Iterable[float],
) -> dict[str, Any]:
    """Report each component and the sum-derived final Prima separately."""
    predictions = np.asarray(y_pred_components, dtype=float)
    if predictions.shape != y_true_components.shape:
        raise ValueError("Component prediction shape does not match component targets.")

    component_metrics = {
        column: compute_regression_metrics(
            y_true_components[column],
            predictions[:, index],
        )
        for index, column in enumerate(y_true_components.columns)
    }
    final_predictions = predictions.sum(axis=1)
    return {
        "components": component_metrics,
        "final_prima": compute_regression_metrics(y_true_total, final_predictions),
    }


def save_grouped_mape_bar_plot(
    categories: Iterable[Any],
    y_true: Iterable[float],
    y_pred: Iterable[float],
    category_name: str,
    output_path: str | Path,
) -> Path:
    """Save final-Prima MAPE grouped by the original category labels."""
    percentage_errors = absolute_percentage_errors(y_true, y_pred)
    plot_data = pd.DataFrame(
        {
            "category": pd.Series(np.asarray(categories, dtype=object), dtype="string").fillna("Missing"),
            "percentage_error": percentage_errors,
        }
    ).dropna(subset=["percentage_error"])
    if plot_data.empty:
        raise ValueError(f"No valid MAPE rows are available for '{category_name}'.")

    grouped = (
        plot_data.groupby("category", sort=True)["percentage_error"]
        .agg(["mean", "size"])
        .sort_index()
    )
    labels = [f"{category} (n={count})" for category, count in zip(grouped.index, grouped["size"])]
    figure_height = max(5.0, 0.3 * len(grouped))
    figure, axis = plt.subplots(figsize=(11, figure_height), constrained_layout=True)
    axis.barh(labels, grouped["mean"], color="steelblue")
    axis.set_title(f"Final Prima MAPE by {category_name}")
    axis.set_xlabel("Mean absolute percentage error (%)")
    axis.set_ylabel(category_name)
    axis.grid(axis="x", alpha=0.2)
    axis.invert_yaxis()

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def create_scored_dataset(
    source_df: pd.DataFrame,
    features: pd.DataFrame,
    component_targets: pd.DataFrame,
    total_target_column: str,
    model: Any,
    train_indices: pd.Index,
    test_indices: pd.Index,
    identity_column: str,
) -> pd.DataFrame:
    """Add full-dataset predictions, percentage errors, and exact split membership."""
    if identity_column not in source_df.columns:
        raise ValueError(f"Identity column '{identity_column}' was not found.")

    scored = source_df.loc[features.index].copy()
    component_predictions = np.asarray(model.predict(features), dtype=float)
    if component_predictions.shape != component_targets.shape:
        raise ValueError("Component prediction shape does not match component targets.")

    split = pd.Series(index=features.index, dtype="string")
    split.loc[train_indices] = "train"
    split.loc[test_indices] = "test"
    if split.isna().any():
        raise ValueError("Some dataset rows were not assigned to train or test.")

    for index, component in enumerate(component_targets.columns):
        scored[f"{component} Prediction"] = component_predictions[:, index]

    final_predictions = component_predictions.sum(axis=1)
    scored["Prediction"] = final_predictions
    scored["Absolute Percent Error"] = absolute_percentage_errors(
        scored[total_target_column],
        final_predictions,
    )
    scored["Dataset Split"] = split

    ordered_columns = [identity_column] + [
        column for column in scored.columns if column != identity_column
    ]
    return scored[ordered_columns]


def save_metrics_log(payload: dict[str, Any], output_path: str | Path) -> Path:
    """Write timestamped run metadata and per-model metrics as readable JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, default=_json_value)
    return path


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__} to JSON.")
