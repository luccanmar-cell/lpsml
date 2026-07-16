from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the dynamic error dashboard for a scored parquet."
    )
    parser.add_argument("dataset", help="Path to a run's scored_dataset.parquet file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Scored parquet was not found: {dataset_path}")
    if importlib.util.find_spec("streamlit") is None:
        raise RuntimeError(
            "Streamlit is required. Install it with 'python -m pip install streamlit'."
        )

    dashboard_path = Path(__file__).with_name("error_dashboard.py")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(dashboard_path),
            "--",
            "--dataset",
            str(dataset_path),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
