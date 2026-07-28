import numpy as np
import pandas as pd
import pytest

from lpsml.data.processing import (
    TARIFF_COVERAGE_FEATURE_PREFIX,
    build_model_dataset,
    split_valid_model_rows,
)
from lpsml.modeling.training import (
    build_joint_strata,
    split_train_test,
    worst_group_mape,
)


def test_model_dataset_one_hot_encodes_only_tariff_coverage_pairs() -> None:
    raw = pd.DataFrame(
        {
            "NroPoliza": ["P1", "P2", "P3"],
            "Pol6TTaCod": ["X", "X", "Y"],
            "Cobertura": [1, 2, 1],
            "Feature": [10.0, 20.0, 30.0],
            "PrimaRC": [100.0, 110.0, 120.0],
            "PrimaCasco": [20.0, 25.0, 30.0],
            "Prima": [120.0, 135.0, 150.0],
        }
    )

    dataset, metadata = build_model_dataset(raw, target_column="Prima")

    pair_prefix = f"{TARIFF_COVERAGE_FEATURE_PREFIX}_"
    pair_columns = {
        column for column in dataset.columns if column.startswith(pair_prefix)
    }
    assert pair_columns == {
        f"{pair_prefix}X__A",
        f"{pair_prefix}X__A2",
        f"{pair_prefix}Y__A",
    }
    assert "Pol6TTaCodEncoded" not in dataset.columns
    assert not any(column.startswith("Cobertura_") for column in dataset.columns)
    assert metadata["reporting_columns"] == ["CoberturaLabel", "Pol6TTaCod"]


def test_data_processing_filters_jointly_underrepresented_pairs() -> None:
    frame = pd.DataFrame(
        {
            "Pol6TTaCod": ["X", "X", "X", "X", "Y", "Y", "Y", "Y"],
            "CoberturaLabel": ["A", "A", "A", "B", "A", "B", "B", "B"],
            "PrimaRC": [100.0] * 8,
            "PrimaCasco": [20.0] * 8,
            "Prima": [120.0] * 8,
            "TariffCoverage_X__A": [1, 1, 1, 0, 0, 0, 0, 0],
            "TariffCoverage_X__B": [0, 0, 0, 1, 0, 0, 0, 0],
            "TariffCoverage_Y__A": [0, 0, 0, 0, 1, 0, 0, 0],
            "TariffCoverage_Y__B": [0, 0, 0, 0, 0, 1, 1, 1],
        }
    )

    clean, doubtful = split_valid_model_rows(
        frame,
        component_target_columns=["PrimaRC", "PrimaCasco"],
        total_target_column="Prima",
        min_pair_count=3,
    )

    assert clean.index.tolist() == [0, 1, 2, 5, 6, 7]
    assert doubtful.index.tolist() == [3, 4]
    assert doubtful["TariffCoveragePairCount"].tolist() == [1, 1]
    assert doubtful["DoubtfulReason"].eq(
        "Rare Pol6TTaCod/Cobertura pair (<3 rows)"
    ).all()
    assert "TariffCoverage_X__A" in clean.columns
    assert "TariffCoverage_Y__B" in clean.columns
    assert "TariffCoverage_X__B" not in clean.columns
    assert "TariffCoverage_Y__A" not in clean.columns


def test_train_test_split_preserves_every_pair_proportionally() -> None:
    pair_counts = {("X", "A"): 20, ("X", "B"): 12, ("Y", "A"): 8}
    pairs = [
        pair
        for pair, count in pair_counts.items()
        for _ in range(count)
    ]
    frame = pd.DataFrame(pairs, columns=["Pol6TTaCod", "CoberturaLabel"])
    features = pd.DataFrame({"Feature": range(len(frame))}, index=frame.index)
    targets = pd.DataFrame({"PrimaRC": range(len(frame))}, index=frame.index)
    strata = build_joint_strata(
        frame,
        ["Pol6TTaCod", "CoberturaLabel"],
    )

    x_train, x_test, _, _ = split_train_test(
        features,
        targets,
        strata,
        test_size=0.25,
        random_state=42,
    )

    train_counts = strata.loc[x_train.index].value_counts()
    test_counts = strata.loc[x_test.index].value_counts()
    assert set(train_counts.index) == set(strata.unique())
    assert set(test_counts.index) == set(strata.unique())
    for label, total in strata.value_counts().items():
        assert abs(test_counts[label] / total - 0.25) <= 1 / total


def test_worst_group_mape_uses_the_highest_pair_mean() -> None:
    actual = pd.DataFrame(
        {
            "PrimaRC": [100.0, 100.0, 200.0, 200.0],
            "PrimaCasco": [0.0, 0.0, 0.0, 0.0],
        }
    )
    predicted = np.array(
        [
            [99.0, 0.0],
            [101.0, 0.0],
            [180.0, 0.0],
            [220.0, 0.0],
        ]
    )
    pairs = pd.Series(["X/A", "X/A", "Y/B", "Y/B"])

    assert np.isclose(worst_group_mape(actual, predicted, pairs), 10.0)


def test_worst_group_mape_rejects_zero_final_premiums() -> None:
    actual = pd.DataFrame({"PrimaRC": [100.0, 0.0]})
    predicted = np.array([[100.0], [1.0]])
    pairs = pd.Series(["X/A", "Y/B"])

    with pytest.raises(ValueError, match="strictly positive"):
        worst_group_mape(actual, predicted, pairs)
