from __future__ import annotations

"""Run hybrid counterfactual scenarios against a trained pricing model.

A scenario selects rows from the original parquet, changes model features before
prediction, and optionally adjusts predicted premium components before computing
the final premium. The exported parquet contains only the selected rows.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from lpsml.modeling.training import load_json_config, predict_premium_components


def parse_args() -> argparse.Namespace:
    """Define the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Apply a counterfactual scenario and export the affected rows.",
        epilog=(
            "The scenario JSON may contain 'selection', 'feature_changes', and "
            "'prediction_adjustments' sections."
        ),
    )
    parser.add_argument("model_path", help="Path to a saved .joblib model.")
    parser.add_argument("dataset", help="Path to the cleaned parquet dataset.")
    parser.add_argument("scenario", help="Path to the counterfactual scenario JSON.")
    parser.add_argument(
        "--config",
        default="configs/model_training.json",
        help="Training configuration containing target and component names.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="artifacts/counterfactual/counterfactual_output.parquet",
        help="Destination parquet path.",
    )
    return parser.parse_args()


def load_scenario(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate a counterfactual scenario."""
    scenario = load_json_config(path)
    if not isinstance(scenario, dict):
        raise ValueError("The scenario must be a JSON object.")

    for section in ("feature_changes", "prediction_adjustments"):
        value = scenario.get(section, [])
        if not isinstance(value, list):
            raise ValueError(f"'{section}' must be a list.")

    selection = scenario.get("selection")
    if selection is not None and not isinstance(selection, dict):
        raise ValueError("'selection' must be an object.")
    return scenario


def build_selection_mask(
    frame: pd.DataFrame,
    expression: dict[str, Any] | None,
) -> pd.Series:
    """Compile a structured selection expression into a boolean pandas mask."""
    if expression is None:
        return pd.Series(True, index=frame.index, dtype=bool)

    logical_keys = [key for key in ("all", "any", "not") if key in expression]
    if logical_keys:
        if len(logical_keys) != 1 or len(expression) != 1:
            raise ValueError("A logical expression must contain only 'all', 'any', or 'not'.")

        logical_key = logical_keys[0]
        if logical_key == "not":
            child = expression["not"]
            if not isinstance(child, dict):
                raise ValueError("'not' must contain one selection expression.")
            return ~build_selection_mask(frame, child)

        children = expression[logical_key]
        if not isinstance(children, list) or not children:
            raise ValueError(f"'{logical_key}' must contain at least one expression.")
        masks = [build_selection_mask(frame, child) for child in children]
        combined = pd.concat(masks, axis=1)
        return combined.all(axis=1) if logical_key == "all" else combined.any(axis=1)

    field = expression.get("field")
    operation = expression.get("op")
    if not isinstance(field, str) or field not in frame.columns:
        raise ValueError(f"Unknown selection field: {field!r}.")
    if not isinstance(operation, str):
        raise ValueError(f"Selection for '{field}' requires an 'op'.")

    values = frame[field]
    if operation == "eq":
        return values == expression.get("value")
    if operation == "ne":
        return values != expression.get("value")
    if operation == "gt":
        return values > expression.get("value")
    if operation == "gte":
        return values >= expression.get("value")
    if operation == "lt":
        return values < expression.get("value")
    if operation == "lte":
        return values <= expression.get("value")
    if operation in {"in", "not_in"}:
        candidates = expression.get("values")
        if not isinstance(candidates, list):
            raise ValueError(f"'{operation}' requires a 'values' list.")
        mask = values.isin(candidates)
        return ~mask if operation == "not_in" else mask
    if operation == "between":
        if "lower" not in expression or "upper" not in expression:
            raise ValueError("'between' requires 'lower' and 'upper'.")
        inclusive_value = expression.get("inclusive", True)
        inclusive = (
            "both"
            if inclusive_value is True
            else "neither"
            if inclusive_value is False
            else inclusive_value
        )
        if inclusive not in {"both", "neither", "left", "right"}:
            raise ValueError("'inclusive' must be true, false, 'left', 'right', or 'both'.")
        return values.between(
            expression["lower"],
            expression["upper"],
            inclusive=inclusive,
        )
    if operation == "is_null":
        return values.isna()
    if operation == "not_null":
        return values.notna()
    raise ValueError(f"Unsupported selection operation: {operation!r}.")


def apply_value_changes(
    frame: pd.DataFrame,
    changes: list[dict[str, Any]],
    field_key: str,
) -> pd.DataFrame:
    """Apply numeric set, additive, multiplicative, or percentage changes."""
    modified = frame.copy()
    for change in changes:
        field = change.get(field_key)
        operation = change.get("op")
        if not isinstance(field, str) or field not in modified.columns:
            raise ValueError(f"Unknown {field_key}: {field!r}.")
        if "value" not in change:
            raise ValueError(f"Change for '{field}' requires a value.")

        try:
            value = float(change["value"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Change value for '{field}' must be numeric.") from exc

        if operation == "set":
            modified[field] = value
        elif operation == "add":
            modified[field] = modified[field] + value
        elif operation == "multiply":
            modified[field] = modified[field] * value
        elif operation == "increase_pct":
            modified[field] = modified[field] * (1.0 + value / 100.0)
        elif operation == "decrease_pct":
            modified[field] = modified[field] * (1.0 - value / 100.0)
        else:
            raise ValueError(f"Unsupported change operation: {operation!r}.")
    return modified


def model_feature_names(
    model: Any,
    source_columns: list[str],
    excluded_columns: list[str],
) -> list[str]:
    """Return the saved model feature names or derive them from the dataset."""
    candidates = [model]
    named_steps = getattr(model, "named_steps", {})
    if "model" in named_steps:
        candidates.append(named_steps["model"])
    for candidate in candidates:
        expected = list(getattr(candidate, "feature_names_in_", []))
        if expected:
            return expected
    excluded = set(excluded_columns)
    return [column for column in source_columns if column not in excluded]


def predict_components(
    model: Any,
    features: pd.DataFrame,
    component_names: list[str],
) -> pd.DataFrame:
    """Predict and label every premium component."""
    predictions = predict_premium_components(model, features)
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
    if predictions.shape != (len(features), len(component_names)):
        raise ValueError(
            "Model prediction shape does not match the configured premium components."
        )
    return pd.DataFrame(predictions, index=features.index, columns=component_names)


def build_counterfactual_frame(
    model: Any,
    source: pd.DataFrame,
    scenario: dict[str, Any],
    component_names: list[str],
    total_name: str,
    excluded_columns: list[str],
) -> pd.DataFrame:
    """Execute one scenario and return only its affected counterfactual rows."""
    mask = build_selection_mask(source, scenario.get("selection"))
    selected = source.loc[mask].copy()
    if selected.empty:
        raise ValueError("The scenario did not select any rows.")

    feature_names = model_feature_names(model, source.columns.tolist(), excluded_columns)
    missing_features = [field for field in feature_names if field not in source.columns]
    if missing_features:
        raise ValueError(f"Model features were not found in the dataset: {missing_features}")

    features = selected[feature_names].apply(pd.to_numeric, errors="coerce")
    if features.isna().any().any():
        missing = features.columns[features.isna().any()].tolist()
        raise ValueError(f"Selected model features contain missing values: {missing}")

    feature_changes = scenario.get("feature_changes", [])
    baseline_components = predict_components(model, features, component_names)
    counterfactual_features = apply_value_changes(features, feature_changes, "field")
    components = predict_components(model, counterfactual_features, component_names)
    components = apply_value_changes(
        components,
        scenario.get("prediction_adjustments", []),
        "component",
    )
    if components.lt(0).any().any():
        raise ValueError("Prediction adjustments produced a negative premium component.")

    result = selected.copy()
    for change in feature_changes:
        field = change["field"]
        result[f"{field}_Counterfactual"] = counterfactual_features[field]
    for component in component_names:
        result[f"{component}_Baseline"] = baseline_components[component]
        result[f"{component}_Counterfactual"] = components[component]
    result[f"{total_name}_Baseline"] = baseline_components.sum(axis=1)
    result[f"{total_name}_Counterfactual"] = components.sum(axis=1)
    return result


def run_counterfactual(
    model_path: str | Path,
    dataset_path: str | Path,
    scenario_path: str | Path,
    config_path: str | Path,
) -> pd.DataFrame:
    """Load all inputs and execute a counterfactual scenario."""
    config = load_json_config(config_path)
    component_names = config.get("component_target_columns", [])
    if not component_names:
        raise ValueError("The config must define 'component_target_columns'.")

    total_name = config.get("target_column", "Prima")
    excluded_columns = [
        *component_names,
        total_name,
        *config.get("drop_columns", ["NroPoliza"]),
    ]
    model = joblib.load(model_path)
    source = pd.read_parquet(dataset_path)
    scenario = load_scenario(scenario_path)
    return build_counterfactual_frame(
        model=model,
        source=source,
        scenario=scenario,
        component_names=component_names,
        total_name=total_name,
        excluded_columns=excluded_columns,
    )


def main() -> None:
    """Run the command-line counterfactual workflow."""
    args = parse_args()
    result = run_counterfactual(
        model_path=args.model_path,
        dataset_path=args.dataset,
        scenario_path=args.scenario,
        config_path=args.config,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    print(f"Saved {len(result)} counterfactual rows to {output_path}")


if __name__ == "__main__":
    main()
