from __future__ import annotations

import argparse
from pathlib import Path

from data_processing_utils import (
    DEFAULT_ACCESSORIES_COLUMN,
    build_model_dataset,
    load_excel_dataset,
    save_dataset_as_parquet,
    split_prima_sum_consistency,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a numeric parquet dataset from the raw LPS Excel tariff file."
    )
    parser.add_argument(
        "filename",
        nargs="?",
        default="tarifacompleto.xlsx",
        help="Path to the raw Excel file. Defaults to tarifacompleto.xlsx.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output parquet path. Defaults to the input filename with a .parquet suffix.",
    )
    parser.add_argument(
        "--doubtful-output",
        help="Rows whose Prima component sum is inconsistent with the total.",
    )
    parser.add_argument(
        "--sum-tolerance",
        type=float,
        default=0.02,
        help="Maximum accepted absolute difference between component sum and Prima.",
    )
    parser.add_argument(
        "--min-pol6tta-count",
        type=int,
        default=50,
        help="Minimum full-dataset frequency required for each Pol6TTaCod category.",
    )
    parser.add_argument(
        "--min-cobertura-count",
        type=int,
        default=50,
        help="Minimum full-dataset frequency required for each Cobertura category.",
    )
    parser.add_argument(
        "--target",
        help="Target column name. Defaults to the final column that starts with Prima.",
    )
    parser.add_argument(
        "--drop-column",
        action="append",
        default=[],
        help="Additional column to drop. Repeat this option for multiple columns.",
    )
    parser.add_argument(
        "--accessories-column",
        default=DEFAULT_ACCESSORIES_COLUMN,
        help="Column containing comma-separated accessory codes.",
    )
    return parser.parse_args()


def default_output_path(input_path: Path) -> Path:
    return input_path.with_suffix(".parquet")


def default_doubtful_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_doubtful{output_path.suffix}")


def main() -> None:
    args = parse_args()
    input_path = Path(args.filename)
    output_path = Path(args.output) if args.output else default_output_path(input_path)
    doubtful_output_path = (
        Path(args.doubtful_output)
        if args.doubtful_output
        else default_doubtful_output_path(output_path)
    )

    raw_df = load_excel_dataset(input_path)
    dataset, metadata = build_model_dataset(
        raw_df,
        target_column=args.target,
        drop_columns=args.drop_column,
        accessories_column=args.accessories_column,
    )
    if "NroPoliza" in raw_df.columns and "NroPoliza" not in dataset.columns:
        dataset["NroPoliza"] = raw_df["NroPoliza"].astype(str).values
    clean_dataset, doubtful_dataset = split_prima_sum_consistency(
        dataset,
        metadata["component_target_columns"],
        metadata["target_column"],
        tolerance=args.sum_tolerance,
        min_pol6tta_count=args.min_pol6tta_count,
        min_cobertura_count=args.min_cobertura_count,
    )
    saved_path = save_dataset_as_parquet(clean_dataset, output_path)
    saved_doubtful_path = save_dataset_as_parquet(
        doubtful_dataset,
        doubtful_output_path,
    )

    print(f"Saved dataset: {saved_path}")
    print(f"Consistent rows: {len(clean_dataset)}")
    print(f"Saved doubtful rows: {saved_doubtful_path}")
    print(f"Doubtful rows: {len(doubtful_dataset)}")
    print(f"Component targets: {metadata['component_target_columns']}")
    print(f"Total target: {metadata['target_column']}")
    print(f"Reporting labels: {metadata['reporting_columns']}")
    print(f"Dropped columns: {metadata['dropped_columns']}")
    print(f"Duplicate modeling rows removed: {metadata['duplicate_rows_removed']}")
    print("\nDataset preview:")
    print(clean_dataset.head())


if __name__ == "__main__":
    main()
