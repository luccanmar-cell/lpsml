import argparse
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from lpsmltest import computemetrics

DEFAULT_DATASET_PATH = "tarifacompleto.parquet"
DEFAULT_TARGET_COLUMN = "Prima"


def train_and_evaluate(
    dataset_path: str | Path,
    target_column: str = DEFAULT_TARGET_COLUMN,
    output_dir: str | Path | None = None,
):
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir) if output_dir is not None else dataset_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(dataset_path)
    df = df.apply(pd.to_numeric, errors="coerce")

    X = df.drop(columns=[target_column, "NroPoliza"], errors="ignore")
    Y = df[target_column]

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    best_model = None
    best_result = None

    parameter_grid = [
        {"n_estimators": 100, "max_depth": None},
        {"n_estimators": 200, "max_depth": None},
        {"n_estimators": 300, "max_depth": None},
        {"n_estimators": 200, "max_depth": 5},
        {"n_estimators": 200, "max_depth": 10},
    ]

    for params in parameter_grid:
        pipeline = make_pipeline(
            StandardScaler(),
            RandomForestRegressor(
                random_state=42,
                n_jobs=-1,
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
            ),
        )

        scores = cross_val_score(
            pipeline,
            X,
            Y,
            cv=cv,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
        )

        mean_mae = -scores.mean()
        print(
            f"n_estimators={params['n_estimators']}, max_depth={params['max_depth']} -> "
            f"CV MAE: {mean_mae:.4f}"
        )

        if best_result is None or mean_mae < best_result:
            best_result = mean_mae
            best_model = pipeline

    print(f"Best cross-validated MAE: {best_result:.4f}")

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    best_model.fit(X_train, Y_train)

    metrics_path = output_dir / f"{dataset_path.stem}_metrics.parquet"
    computemetrics(
        best_model,
        X_test,
        Y_test,
        X_train,
        dataset_path=dataset_path,
        metrics_output_path=metrics_path,
        output_dir=output_dir,
    )

    return best_model, {
        "dataset_path": dataset_path,
        "metrics_path": metrics_path,
        "target_column": target_column,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a model from a prepared parquet dataset.")
    parser.add_argument(
        "dataset",
        nargs="?",
        default=DEFAULT_DATASET_PATH,
        help="Path to the prepared parquet dataset.",
    )
    parser.add_argument(
        "--target-column",
        default=DEFAULT_TARGET_COLUMN,
        help="Target column to use for training.",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Folder where metrics and plots will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_and_evaluate(args.dataset, target_column=args.target_column, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
