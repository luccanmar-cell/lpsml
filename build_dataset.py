from __future__ import annotations

import argparse
from pathlib import Path

from lpsml.data.processing import (
    DEFAULT_ACCESSORIES_COLUMN,
    build_model_dataset,
    load_excel_dataset,
    save_dataset_as_parquet,
    split_valid_model_rows,
)

SUPPORTED_WORKBOOK_SUFFIXES = {".xlsx", ".xlsm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a numeric parquet dataset from the raw LPS Excel tariff file."
    )
    parser.add_argument(
        "filename",
        nargs="?",
        help=(
            "Path to the raw Excel workbook. When omitted, uses the only "
            "supported workbook under data/raw."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        help=(
            "Output parquet path. Defaults to data/processed/<input name>.parquet."
        ),
    )
    parser.add_argument(
        "--doubtful-output",
        help="Rows that fail premium, coverage, or category-frequency checks.",
    )
    parser.add_argument(
        "--sum-tolerance",
        type=float,
        default=0.02,
        help="Maximum accepted absolute difference between component sum and Prima.",
    )
    parser.add_argument(
        "--min-pair-count",
        type=int,
        default=50,
        help=(
            "Minimum number of otherwise valid rows required for each "
            "Pol6TTaCod/Cobertura pair."
        ),
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


def resolve_input_path(
    filename: str | None,
    raw_directory: str | Path = "data/raw",
) -> Path:
    """Resolve an explicit workbook or discover the only workbook in data/raw."""
    if filename:
        return Path(filename)

    directory = Path(raw_directory)
    candidates = (
        sorted(
            (
                path
                for path in directory.iterdir()
                if path.is_file()
                and path.suffix.lower() in SUPPORTED_WORKBOOK_SUFFIXES
            ),
            key=lambda path: path.name.lower(),
        )
        if directory.is_dir()
        else []
    )
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"No Excel workbook was found in {directory}. "
            "Add one .xlsx or .xlsm file, or provide an explicit input path."
        )
    names = ", ".join(path.name for path in candidates)
    raise ValueError(
        f"Expected one Excel workbook in {directory}, found {len(candidates)}: "
        f"{names}. Provide an explicit input path."
    )


def default_output_path(input_path: Path) -> Path:
    return Path("data/processed") / f"{input_path.stem}.parquet"


def default_doubtful_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_doubtful{output_path.suffix}")


def main() -> None:
    args = parse_args()
    input_path = resolve_input_path(args.filename)
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
    clean_dataset, doubtful_dataset = split_valid_model_rows(
        dataset,
        metadata["component_target_columns"],
        metadata["target_column"],
        tolerance=args.sum_tolerance,
        min_pair_count=args.min_pair_count,
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
