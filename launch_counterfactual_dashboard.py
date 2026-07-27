from __future__ import annotations

"""Launch the Streamlit counterfactual-impact dashboard."""

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the counterfactual dashboard for an output Parquet."
    )
    parser.add_argument("dataset", help="Path to a counterfactual output Parquet.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Counterfactual Parquet was not found: {dataset_path}")
    if importlib.util.find_spec("streamlit") is None:
        raise RuntimeError(
            "Streamlit is required. Install it with 'python -m pip install streamlit'."
        )

    repository_root = Path(__file__).resolve().parent
    dashboard_path = repository_root / "lpsml" / "dashboards" / "counterfactual.py"
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
        cwd=repository_root,
    )


if __name__ == "__main__":
    main()
