from __future__ import annotations

import argparse
from pathlib import Path

from data_processing_utils import (
    DEFAULT_ACCESSORIES_COLUMN,
    build_model_dataset,
    load_excel_dataset,
    save_dataset_as_parquet,
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


def build_dataset_file(
    input_path: str | Path,
    output_path: str | Path | None = None,
    target_column: str | None = None,
    drop_columns: list[str] | None = None,
    accessories_column: str = DEFAULT_ACCESSORIES_COLUMN,
) -> tuple[Path, dict[str, object]]:
    input_path = Path(input_path)
    output_path = Path(output_path) if output_path else default_output_path(input_path)

    raw_df = load_excel_dataset(input_path)
    dataset, metadata = build_model_dataset(
        raw_df,
        target_column=target_column,
        drop_columns=drop_columns or [],
        accessories_column=accessories_column,
    )
    if "NroPoliza" in raw_df.columns and "NroPoliza" not in dataset.columns:
        dataset["NroPoliza"] = raw_df["NroPoliza"].astype(str).values
    saved_path = save_dataset_as_parquet(dataset, output_path)

    print(f"Saved dataset: {saved_path}")
    print(f"Rows: {metadata['row_count']} | Columns: {metadata['column_count']}")
    print(f"Target: {metadata['target_column']}")
    print(f"Dropped leakage columns: {metadata['dropped_columns']}")
    print("\nDataset preview:")
    print(dataset.head())

    return saved_path, metadata


def main() -> None:
    args = parse_args()
    build_dataset_file(
        input_path=args.filename,
        output_path=args.output,
        target_column=args.target,
        drop_columns=args.drop_column,
        accessories_column=args.accessories_column,
    )


if __name__ == "__main__":
    main()
