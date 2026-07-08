from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_ACCESSORIES_COLUMN = "Accesorios"


def load_excel_dataset(filename: str | Path) -> pd.DataFrame:
    """Read the raw Excel workbook into a DataFrame."""
    return pd.read_excel(filename, engine="openpyxl")


def infer_target_column(columns: Iterable[str], requested_target: str | None = None) -> str:
    """Find the target column, defaulting to the final column that starts with Prima."""
    column_list = list(columns)

    if requested_target:
        if requested_target not in column_list:
            raise ValueError(f"Target column '{requested_target}' was not found.")
        return requested_target

    prima_columns = [column for column in column_list if column.lower().startswith("prima")]
    if not prima_columns:
        raise ValueError("No target was provided and no column starting with 'Prima' was found.")

    return prima_columns[-1]


def default_leakage_columns(columns: Iterable[str], target_column: str) -> list[str]:
    """Drop intermediate Prima columns while keeping the final Prima target."""
    return [
        column
        for column in columns
        if column != target_column and column.lower().startswith("prima")
    ]


def split_accessory_codes(value: object) -> list[str]:
    """Turn values like '16, 46' into clean accessory-code tokens."""
    if pd.isna(value):
        return []

    tokens: list[str] = []
    for raw_token in str(value).split(","):
        token = raw_token.strip()
        if not token:
            continue

        numeric_token = pd.to_numeric(token, errors="coerce")
        if pd.notna(numeric_token) and float(numeric_token).is_integer():
            token = str(int(numeric_token))

        tokens.append(token)

    return tokens


def one_hot_encode_accessories(
    df: pd.DataFrame,
    accessories_column: str = DEFAULT_ACCESSORIES_COLUMN,
) -> pd.DataFrame:
    """Replace Accesorios with one binary column per accessory code."""
    if accessories_column not in df.columns:
        return df.copy()

    accessory_lists = df[accessories_column].apply(split_accessory_codes)
    all_codes = sorted(
        {code for row_codes in accessory_lists for code in row_codes},
        key=lambda code: (not code.isdigit(), int(code) if code.isdigit() else code),
    )

    encoded = pd.DataFrame(index=df.index)
    for code in all_codes:
        # Each feature indicates whether that row contains the accessory code.
        encoded[f"{accessories_column}_{code}"] = accessory_lists.apply(
            lambda row_codes, code=code: int(code in row_codes)
        )

    return pd.concat([df.drop(columns=[accessories_column]), encoded], axis=1)


def convert_numeric_like_columns(df: pd.DataFrame, exclude_columns: Iterable[str]) -> pd.DataFrame:
    """Convert object columns to numbers when every non-empty value is numeric."""
    cleaned = df.copy()
    excluded = set(exclude_columns)

    for column in cleaned.columns:
        if column in excluded or not pd.api.types.is_object_dtype(cleaned[column]):
            continue

        converted = pd.to_numeric(cleaned[column], errors="coerce")
        has_same_non_null_values = converted.notna().sum() == cleaned[column].notna().sum()
        if has_same_non_null_values:
            cleaned[column] = converted

    return cleaned


def ordinal_encode_categoricals(df: pd.DataFrame, exclude_columns: Iterable[str]) -> pd.DataFrame:
    """Convert remaining categorical columns into stable integer category codes."""
    encoded = df.copy()
    excluded = set(exclude_columns)

    for column in encoded.columns:
        if column in excluded:
            continue

        is_categorical = (
            pd.api.types.is_object_dtype(encoded[column])
            or pd.api.types.is_string_dtype(encoded[column])
            or isinstance(encoded[column].dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(encoded[column])
        )
        if not is_categorical:
            continue

        non_null_values = encoded[column].dropna().astype(str)
        categories = sorted(non_null_values.unique())
        encoded[column] = pd.Categorical(
            encoded[column].astype("string"),
            categories=categories,
        ).codes

    return encoded


def fill_missing_feature_values(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Fill feature nulls with 0 while keeping target values as-is."""
    filled = df.copy()
    feature_columns = [column for column in filled.columns if column != target_column]
    filled[feature_columns] = filled[feature_columns].fillna(0)
    return filled


def build_model_dataset(
    raw_df: pd.DataFrame,
    target_column: str | None = None,
    drop_columns: Iterable[str] | None = None,
    accessories_column: str = DEFAULT_ACCESSORIES_COLUMN,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Create a numeric dataset ready for model training."""
    target = infer_target_column(raw_df.columns, target_column)
    leakage_columns = default_leakage_columns(raw_df.columns, target)
    extra_drop_columns = list(drop_columns or [])
    columns_to_drop = [
        column
        for column in dict.fromkeys([*leakage_columns, *extra_drop_columns])
        if column in raw_df.columns and column != target
    ]

    df = raw_df.drop(columns=columns_to_drop).dropna(subset=[target]).copy()
    df = one_hot_encode_accessories(df, accessories_column=accessories_column)
    df = convert_numeric_like_columns(df, exclude_columns=[target])
    df = ordinal_encode_categoricals(df, exclude_columns=[target])
    df = fill_missing_feature_values(df, target_column=target)
    df = df[[column for column in df.columns if column != target] + [target]]

    metadata = {
        "target_column": target,
        "dropped_columns": columns_to_drop,
        "row_count": len(df),
        "column_count": len(df.columns),
    }
    return df, metadata


def save_dataset_as_parquet(df: pd.DataFrame, output_path: str | Path) -> Path:
    """Persist the processed dataset in a compact training-friendly format."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path
