from __future__ import annotations

"""The script performs four main steps:
1. Loads the model and the processed dataset.
2. Selects the requested rows.
3. Applies the requested changes to the feature values.
4. Compares baseline and modified predictions and optionally exports the results.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from training_utils import load_json_config, load_training_data


def parse_args() -> argparse.Namespace:
    """Define the command-line interface and explain how to run the script.

    Use this function when you want to call the script from a terminal or from another
    automation workflow. The arguments let you choose the model, dataset, rows to inspect,
    and the feature changes to apply.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Load a trained model, modify selected rows, compare baseline vs. "
            "modified predictions, and optionally export the result."
        )
    )
    parser.add_argument("model_path", help="Path to a saved .joblib model.")
    parser.add_argument("dataset", help="Processed parquet dataset used to train the model.")
    parser.add_argument(
        "--config",
        default="model_training_config.json",
        help="JSON configuration with training target and drop-column settings.",
    )
    parser.add_argument(
        "--rows",
        nargs="+",
        type=int,
        default=[0],
        help="One or more row positions to inspect and modify (0-based).",
    )
    parser.add_argument(
        "--modify",
        action="append",
        default=[],
        help=(
            "Feature change to apply. Examples: 'ValorAsegurado=500000' or "
            "'5:ValorAsegurado=500000'."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON file path for the comparison report.",
    )
    parser.add_argument(
        "--parquet-output",
        default=None,
        help="Optional parquet file path for the modified rows export.",
    )
    parser.add_argument(
        "--identity-column",
        default=None,
        help="Column to preserve in the parquet export; defaults to the config value.",
    )
    return parser.parse_args()


def parse_modification_specs(specs: list[str]) -> list[tuple[int | None, str, Any]]:
    """Convert user-friendly change instructions into structured values.

    Examples:
    - "ValorAsegurado=500000" changes the named column for all selected rows.
    - "3:ValorAsegurado=500000" changes only row 3 in the selected set.
    """
    parsed: list[tuple[int | None, str, Any]] = []
    pattern = re.compile(r"^(?:(?P<row>\d+):)?(?P<column>.+?)=(?P<value>.+)$")

    for spec in specs:
        match = pattern.match(spec.strip())
        if match is None:
            raise ValueError(
                f"Invalid modification spec '{spec}'. Use 'column=value' or 'row:column=value'."
            )

        row = int(match.group("row")) if match.group("row") is not None else None
        column = match.group("column").strip()
        raw_value = match.group("value").strip()

        try:
            value: Any = float(raw_value)
        except ValueError:
            value = raw_value

        parsed.append((row, column, value))

    return parsed


def apply_modifications(frame: pd.DataFrame, specs: list[tuple[int | None, str, Any]]) -> pd.DataFrame:
    """Apply the requested changes to a copy of the dataframe."""
    modified = frame.copy()
    for row_index, column, value in specs:
        if row_index is None:
            modified.loc[:, column] = value
        else:
            if row_index in modified.index:
                modified.loc[row_index, column] = value
            elif row_index < len(modified):
                modified.iloc[row_index, modified.columns.get_loc(column)] = value
            else:
                raise IndexError(
                    f"Row index {row_index} is out of bounds for a frame with {len(modified)} rows."
                )
    return modified


def align_features_to_model(frame: pd.DataFrame, model: Any) -> pd.DataFrame:
    """Ensure the input columns match the trained model's expected feature order."""
    expected_columns = list(getattr(model, "feature_names_in_", []))
    if not expected_columns:
        return frame.copy()

    aligned = pd.DataFrame(index=frame.index)
    for column in expected_columns:
        aligned[column] = frame[column] if column in frame.columns else 0.0
    return aligned


def build_prediction_report(
    model_path: str | Path,
    dataset_path: str | Path,
    config_path: str | Path,
    row_positions: list[int],
    modification_specs: list[tuple[int | None, str, Any]],
    identity_column: str | None = None,
) -> dict[str, Any]:
    """Loads the model and data, applies the
    requested changes, predicts both versions, and returns the differences in a report
    structure that can be printed or exported.
    """
    model_path = Path(model_path)
    dataset_path = Path(dataset_path)
    config_path = Path(config_path)

    config = load_json_config(config_path)
    total_target_column = config.get("target_column", "Prima")
    component_target_columns = config.get("component_target_columns", [])
    drop_columns = config.get("drop_columns", ["NroPoliza"])

    features, _ = load_training_data(
        dataset_path,
        target_columns=component_target_columns,
        total_target_column=total_target_column,
        drop_columns=drop_columns,
    )

    if not row_positions:
        raise ValueError("At least one row position must be provided via --rows.")
    if any(position < 0 or position >= len(features) for position in row_positions):
        raise ValueError(
            f"Rows must be between 0 and {len(features) - 1} for the current dataset."
        )

    model = joblib.load(model_path)
    selected_rows = features.iloc[row_positions].copy()
    baseline_features = align_features_to_model(selected_rows, model)
    modified_features = apply_modifications(selected_rows.copy(), modification_specs)
    modified_features = align_features_to_model(modified_features, model)

    baseline_predictions = np.asarray(model.predict(baseline_features))
    modified_predictions = np.asarray(model.predict(modified_features))

    if baseline_predictions.ndim == 1:
        baseline_predictions = baseline_predictions.reshape(-1, 1)
    if modified_predictions.ndim == 1:
        modified_predictions = modified_predictions.reshape(-1, 1)

    baseline_rows = []
    modified_rows = []
    deltas = []
    for index, row_position in enumerate(row_positions):
        baseline_row = baseline_predictions[index]
        modified_row = modified_predictions[index]
        delta_row = modified_row - baseline_row
        baseline_rows.append(baseline_row.tolist())
        modified_rows.append(modified_row.tolist())
        deltas.append(delta_row.tolist())

    if component_target_columns:
        prediction_columns = [f"prediction_{column}" for column in component_target_columns]
    else:
        prediction_columns = [f"prediction_{index}" for index in range(baseline_predictions.shape[1])]

    report = {
        "model_path": str(model_path),
        "dataset_path": str(dataset_path),
        "rows": row_positions,
        "modifications": [
            {"row": row_index, "column": column, "value": value}
            for row_index, column, value in modification_specs
        ],
        "baseline_predictions": baseline_rows,
        "modified_predictions": modified_rows,
        "prediction_deltas": deltas,
        "prediction_columns": prediction_columns,
    }

    if identity_column is None:
        identity_column = config.get("identity_column", "NroPoliza")

    if identity_column:
        full_dataset = pd.read_parquet(dataset_path)
        selected_index = features.index[row_positions]
        export_frame = full_dataset.loc[selected_index].copy()
        export_frame["baseline_prediction"] = [row[0] if len(row) == 1 else row for row in baseline_rows]
        export_frame["modified_prediction"] = [row[0] if len(row) == 1 else row for row in modified_rows]
        export_frame["prediction_delta"] = [row[0] if len(row) == 1 else row for row in deltas]
        if identity_column in export_frame.columns:
            export_frame = export_frame[[identity_column, *[col for col in export_frame.columns if col != identity_column]]]
        report["export_frame"] = export_frame

    return report


def main() -> None:
    """Run the sensitivity-analysis"""
    args = parse_args()
    specs = parse_modification_specs(args.modify)
    report = build_prediction_report(
        model_path=args.model_path,
        dataset_path=args.dataset,
        config_path=args.config,
        row_positions=args.rows,
        modification_specs=specs,
        identity_column=args.identity_column,
    )

    print(json.dumps(report, indent=2, default=str))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"Saved JSON report to {output_path}")

    if args.parquet_output:
        export_frame = report.get("export_frame")
        if export_frame is None:
            raise ValueError("Parquet export requires the dataset identity column to be available.")
        parquet_path = Path(args.parquet_output)
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        export_frame.to_parquet(parquet_path, index=False)
        print(f"Saved modified rows parquet to {parquet_path}")


if __name__ == "__main__":
    main()
