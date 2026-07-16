from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from metrics_utils import (
    compute_multioutput_metrics,
    create_scored_dataset,
    save_grouped_mape_bar_plot,
    save_metrics_log,
)
from training_utils import (
    load_json_config,
    load_training_data,
    search_models,
    split_train_test,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select and train a scikit-learn regression model."
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        help="Processed parquet dataset. Defaults to dataset_path from the config.",
    )
    parser.add_argument(
        "--config",
        default="model_training_config.json",
        help="JSON file defining the split, CV, models, scalers, and search spaces.",
    )
    parser.add_argument(
        "--search",
        choices=["grid", "optuna"],
        help="Override search.method from the JSON configuration.",
    )
    return parser.parse_args()


def print_search_results(results: list[dict[str, Any]]) -> None:
    print("\nCross-validation results (training split only):")
    for result in results:
        print(
            f"  {result['model']} | multi-output={result['multioutput_strategy']} | "
            f"scaler={result['scaler']} | "
            f"balanced MSE={result['cv_balanced_mse']:.4f} | "
            f"params={result['best_params']}"
        )


def best_result_per_model_type(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep the lowest-CV balanced-MSE configuration for each model type."""
    best_by_type: dict[str, dict[str, Any]] = {}
    for result in results:
        current = best_by_type.get(result["model_type"])
        if (
            current is None
            or result["cv_balanced_mse"] < current["cv_balanced_mse"]
        ):
            best_by_type[result["model_type"]] = result
    return list(best_by_type.values())


def print_test_metrics(model_name: str, metrics: dict[str, Any]) -> None:
    print(f"\nTest metrics for {model_name}:")
    for target_name, target_metrics in [
        *metrics["components"].items(),
        ("Final Prima", metrics["final_prima"]),
    ]:
        mape = target_metrics["mape_percent"]
        mape_text = f"{mape:.2f}%" if mape is not None else "undefined"
        print(
            f"  {target_name}: MSE={target_metrics['mse']:.4f}, "
            f"MAE={target_metrics['mae']:.4f}, "
            f"MAPE={mape_text} "
            f"({target_metrics['zero_target_rows']} zero-target rows excluded)"
        )


def main() -> None:
    args = parse_args()
    config = load_json_config(args.config)
    if args.search:
        config.setdefault("search", {})["method"] = args.search

    dataset_path = args.dataset or config.get("dataset_path")
    if not dataset_path:
        raise ValueError("Provide a dataset argument or set dataset_path in the config.")

    total_target_column = config.get("target_column", "Prima")
    component_target_columns = config["component_target_columns"]
    identity_column = config.get("identity_column", "NroPoliza")
    random_state = int(config.get("random_state", 42))
    features, target = load_training_data(
        Path(dataset_path),
        target_columns=component_target_columns,
        total_target_column=total_target_column,
        drop_columns=config.get("drop_columns", ["NroPoliza"]),
    )
    x_train, x_test, y_train, y_test = split_train_test(
        features,
        target,
        test_size=float(config.get("test_size", 0.2)),
        random_state=random_state,
    )

    best_model, best_result, results = search_models(x_train, y_train, config)
    print_search_results(results)
    print(
        f"\nSelected by CV balanced MSE: {best_result['model']} with "
        f"scaler={best_result['scaler']} and params={best_result['best_params']}"
    )

    run_time = datetime.now().astimezone()
    run_id = run_time.strftime("%Y%m%d_%H%M%S_%f")
    run_directory = Path(config.get("output_directory", "training_runs")) / run_id
    run_directory.mkdir(parents=True, exist_ok=False)

    source_df = pd.read_parquet(dataset_path)
    mape_group_columns = config.get(
        "mape_group_columns",
        {
            "Cobertura": "CoberturaLabel",
            "Pol6TTaCod": "Pol6TTaCod",
        },
    )
    model_logs: list[dict[str, Any]] = []
    for result in best_result_per_model_type(results):
        model = result["estimator"]
        predictions = model.predict(x_test)
        metrics = compute_multioutput_metrics(
            y_test,
            predictions,
            source_df.loc[y_test.index, total_target_column],
        )
        print_test_metrics(result["model"], metrics)

        model_directory = run_directory / result["model_type"]
        model_directory.mkdir(parents=True, exist_ok=False)
        model_path = model_directory / "model.joblib"
        joblib.dump(model, model_path)
        final_predictions = predictions.sum(axis=1)
        final_actual = source_df.loc[y_test.index, total_target_column]
        grouped_plot_paths = {
            display_name: save_grouped_mape_bar_plot(
                source_df.loc[y_test.index, source_column],
                final_actual,
                final_predictions,
                display_name,
                model_directory / f"mape_by_{display_name.lower()}.png",
            )
            for display_name, source_column in mape_group_columns.items()
        }
        model_log = {
            "run_timestamp": run_time.isoformat(),
            "model": result["model"],
            "model_type": result["model_type"],
            "multioutput_strategy": result["multioutput_strategy"],
            "scaler": result["scaler"],
            "search_trials": result["search_trials"],
            "cv_balanced_mse": result["cv_balanced_mse"],
            "best_params": result["best_params"],
            "test_metrics": metrics,
            "model_path": model_path,
            "grouped_mape_plot_paths": grouped_plot_paths,
        }
        model_metrics_path = model_directory / "metrics.json"
        save_metrics_log(model_log, model_metrics_path)
        model_log["metrics_path"] = model_metrics_path
        model_logs.append(model_log)

    scored_dataset = create_scored_dataset(
        source_df,
        features,
        target,
        total_target_column,
        best_model,
        train_indices=x_train.index,
        test_indices=x_test.index,
        identity_column=identity_column,
    )
    scored_dataset_path = run_directory / "scored_dataset.parquet"
    scored_dataset.to_parquet(scored_dataset_path, index=False)
    scored_excel_path = run_directory / "scored_dataset.xlsx"
    scored_dataset.sort_values(
        "Absolute Percent Error",
        ascending=False,
        na_position="last",
    ).to_excel(scored_excel_path, index=False, engine="openpyxl")

    overall_model_log = next(
        model_log
        for model_log in model_logs
        if model_log["model_type"] == best_result["model_type"]
    )
    metrics_path = run_directory / "run_summary.json"
    save_metrics_log(
        {
            "run_timestamp": run_time.isoformat(),
            "dataset_path": str(dataset_path),
            "target_column": total_target_column,
            "component_target_columns": component_target_columns,
            "identity_column": identity_column,
            "search_method": config.get("search", {}).get("method", "grid"),
            "random_state": random_state,
            "train_rows": len(x_train),
            "test_rows": len(x_test),
            "overall_best_model_type": best_result["model_type"],
            "overall_best_multioutput_strategy": best_result["multioutput_strategy"],
            "overall_best_scaler": best_result["scaler"],
            "overall_best_params": best_result["best_params"],
            "overall_best_cv_balanced_mse": best_result["cv_balanced_mse"],
            "overall_best_model_path": overall_model_log["model_path"],
            "scored_dataset_path": scored_dataset_path,
            "scored_excel_path": scored_excel_path,
            "models": model_logs,
        },
        metrics_path,
    )

    print(f"\nSaved run artifacts: {run_directory}")
    print(f"Saved test metrics: {metrics_path}")
    print(f"Saved full scored dataset: {scored_dataset_path}")
    print(f"Saved sorted scored Excel dataset: {scored_excel_path}")


if __name__ == "__main__":
    main()
