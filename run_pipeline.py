from __future__ import annotations

import argparse
from pathlib import Path

from build_dataset import build_dataset_file, default_output_path
from lpsmltraining import train_and_evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full LPSML workflow from a raw Excel file with a single command."
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        default="tarifacompleto.xlsx", #Pone el nombre del archivo de Excel que deseas procesar por defecto
        help="Path to the raw Excel file to process.",
    )
    parser.add_argument(
        "--output",
        help="Optional parquet output path for the prepared dataset.",
    )
    parser.add_argument(
        "--target",
        help="Target column name. Defaults to the final Prima column.",
    )
    parser.add_argument(
        "--drop-column",
        action="append",
        default=[],
        help="Additional column to drop. Repeat for multiple columns.",
    )
    parser.add_argument(
        "--accessories-column",
        default="Accesorios",
        help="Column containing the accessory codes.",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory where metrics and plots will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_file)
    output_path = Path(args.output) if args.output else default_output_path(input_path)

    print(f"Building dataset from {input_path}...")
    build_dataset_file(
        input_path=input_path,
        output_path=output_path,
        target_column=args.target,
        drop_columns=args.drop_column,
        accessories_column=args.accessories_column,
    )

    print(f"Training model from {output_path}...")
    train_and_evaluate(output_path, target_column=args.target or "Prima", output_dir=args.output_dir)


if __name__ == "__main__":
    main()
