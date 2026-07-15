from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_ACCESSORIES_COLUMN = "Accesorios"
COBERTURA_LABELS = {
    1: "A",
    2: "A2",
    3: "B1",
    4: "B",
    5: "B2",
    6: "XB",
    7: "C1",
    8: "C4",
    9: "C1+",
    10: "C",
    11: "C2",
    12: "C3",
    13: "D2C",
    14: "D2H",
    15: "D6%",
    16: "D5%",
    17: "D4%",
    18: "D3%",
    19: "D5",
    20: "D6",
    21: "D2%",
    22: "D1%",
    23: "D54",
    24: "D32",
    25: "D2I",
    26: "D36",
    27: "D2",
    28: "D3I",
    29: "D3",
    30: "D4I",
    31: "D4",
    32: "D",
    33: "D1",
}


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


def infer_component_target_columns(
    columns: Iterable[str],
    total_target_column: str,
) -> list[str]:
    """Return the intermediate Prima columns whose sum should produce the total."""
    return [
        column
        for column in columns
        if column != total_target_column and column.lower().startswith("prima")
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


def cobertura_label(value: object) -> str:
    """Translate the numeric Cobertura category into its original policy code."""
    if pd.isna(value):
        return "Missing"

    numeric_value = pd.to_numeric(value, errors="coerce")
    if pd.notna(numeric_value) and float(numeric_value).is_integer():
        category_number = int(numeric_value)
        if category_number in COBERTURA_LABELS:
            return COBERTURA_LABELS[category_number]

    return str(value)


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


def fill_missing_feature_values(
    df: pd.DataFrame,
    target_columns: Iterable[str],
) -> pd.DataFrame:
    """Fill feature nulls with 0 while keeping all target values as-is."""
    filled = df.copy()
    target_set = set(target_columns)
    feature_columns = [column for column in filled.columns if column not in target_set]
    filled[feature_columns] = filled[feature_columns].fillna(0)
    return filled


def build_model_dataset(
    raw_df: pd.DataFrame,
    target_column: str | None = None,
    drop_columns: Iterable[str] | None = None,
    accessories_column: str = DEFAULT_ACCESSORIES_COLUMN,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Create a numeric dataset with component and total Prima targets retained."""
    total_target = infer_target_column(raw_df.columns, target_column)
    component_targets = infer_component_target_columns(raw_df.columns, total_target)
    target_columns = [*component_targets, total_target]
    extra_drop_columns = list(drop_columns or [])
    columns_to_drop = [
        column
        for column in dict.fromkeys(extra_drop_columns)
        if column in raw_df.columns and column not in target_columns
    ]

    df = raw_df.drop(columns=columns_to_drop).dropna(subset=target_columns).copy()
    df = one_hot_encode_accessories(df, accessories_column=accessories_column)
    reporting_columns: list[str] = []
    if "Cobertura" in df.columns:
        df["CoberturaLabel"] = df["Cobertura"].map(cobertura_label).astype("string")
        reporting_columns.append("CoberturaLabel")

    if "Pol6TTaCod" in df.columns:
        original_values = df["Pol6TTaCod"].astype("string").fillna("Missing")
        categories = sorted(original_values.unique())
        df["Pol6TTaCodEncoded"] = pd.Categorical(
            original_values,
            categories=categories,
        ).codes
        df["Pol6TTaCod"] = original_values
        reporting_columns.append("Pol6TTaCod")

    df = convert_numeric_like_columns(
        df,
        exclude_columns=[*target_columns, *reporting_columns, "Pol6TTaCod"],
    )
    df = ordinal_encode_categoricals(
        df,
        exclude_columns=[*target_columns, *reporting_columns],
    )
    df = fill_missing_feature_values(df, target_columns=target_columns)
    feature_columns = [column for column in df.columns if column not in target_columns]
    df = df[[*feature_columns, *target_columns]]

    metadata = {
        "target_column": total_target,
        "component_target_columns": component_targets,
        "target_columns": target_columns,
        "reporting_columns": reporting_columns,
        "dropped_columns": columns_to_drop,
        "row_count": len(df),
        "column_count": len(df.columns),
    }
    return df, metadata


def split_prima_sum_consistency(
    df: pd.DataFrame,
    component_target_columns: Iterable[str],
    total_target_column: str,
    tolerance: float = 0.02,
    min_pol6tta_count: int = 50,
    min_cobertura_count: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate rows that satisfy target, coverage, and category-frequency checks."""
    if min_pol6tta_count < 1:
        raise ValueError("min_pol6tta_count must be at least 1.")
    if min_cobertura_count < 1:
        raise ValueError("min_cobertura_count must be at least 1.")

    component_columns = list(component_target_columns)
    difference = df[total_target_column] - df[component_columns].sum(axis=1)
    sum_consistent = np.abs(difference.to_numpy(dtype=float)) <= tolerance + 1e-9
    if "CoberturaLabel" in df.columns:
        cobertura_present = df["CoberturaLabel"].ne("Missing").to_numpy()
    else:
        cobertura_present = df["Cobertura"].notna().to_numpy()

    pol6tta_counts = df["Pol6TTaCod"].value_counts(dropna=False)
    row_pol6tta_counts = df["Pol6TTaCod"].map(pol6tta_counts).to_numpy(dtype=int)
    pol6tta_sufficient = row_pol6tta_counts >= min_pol6tta_count
    cobertura_counts = df["CoberturaLabel"].value_counts(dropna=False)
    row_cobertura_counts = df["CoberturaLabel"].map(cobertura_counts).to_numpy(dtype=int)
    cobertura_sufficient = row_cobertura_counts >= min_cobertura_count
    clean_mask = (
        sum_consistent
        & cobertura_present
        & cobertura_sufficient
        & pol6tta_sufficient
    )

    clean_df = df.loc[clean_mask].copy()
    doubtful_df = df.loc[~clean_mask].copy()
    doubtful_df["PrimaSumDifference"] = difference.loc[~clean_mask]
    doubtful_df["Pol6TTaCodCount"] = row_pol6tta_counts[~clean_mask]
    doubtful_df["CoberturaCount"] = row_cobertura_counts[~clean_mask]
    doubtful_df["DoubtfulReason"] = [
        "; ".join(
            reason
            for condition, reason in [
                (not sum_is_consistent, "Prima component sum mismatch"),
                (not has_cobertura, "Missing Cobertura"),
                (
                    has_cobertura and not has_sufficient_cobertura,
                    f"Rare Cobertura (<{min_cobertura_count} rows)",
                ),
                (
                    not has_sufficient_pol6tta,
                    f"Rare Pol6TTaCod (<{min_pol6tta_count} rows)",
                ),
            ]
            if condition
        )
        for (
            sum_is_consistent,
            has_cobertura,
            has_sufficient_cobertura,
            has_sufficient_pol6tta,
        ) in zip(
            sum_consistent[~clean_mask],
            cobertura_present[~clean_mask],
            cobertura_sufficient[~clean_mask],
            pol6tta_sufficient[~clean_mask],
        )
    ]
    return clean_df, doubtful_df


def save_dataset_as_parquet(df: pd.DataFrame, output_path: str | Path) -> Path:
    """Persist the processed dataset in a compact training-friendly format."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path
