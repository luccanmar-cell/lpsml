from pathlib import Path

import pytest

from build_dataset import resolve_input_path


def test_discovers_the_only_supported_workbook(tmp_path: Path) -> None:
    workbook = tmp_path / "portfolio.xlsx"
    workbook.touch()
    (tmp_path / ".gitkeep").touch()
    (tmp_path / "notes.txt").touch()

    assert resolve_input_path(None, tmp_path) == workbook


def test_discovery_requires_a_workbook(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No Excel workbook"):
        resolve_input_path(None, tmp_path)


def test_discovery_rejects_multiple_workbooks(tmp_path: Path) -> None:
    (tmp_path / "first.xlsx").touch()
    (tmp_path / "second.xlsm").touch()

    with pytest.raises(ValueError, match="found 2"):
        resolve_input_path(None, tmp_path)


def test_explicit_input_does_not_depend_on_raw_directory(tmp_path: Path) -> None:
    explicit = tmp_path / "elsewhere.xlsx"

    assert resolve_input_path(str(explicit), tmp_path / "missing") == explicit
