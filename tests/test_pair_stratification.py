import pandas as pd

from lpsml.data.processing import split_valid_model_rows
from lpsml.modeling.training import build_joint_strata, split_train_test


def test_data_processing_filters_jointly_underrepresented_pairs() -> None:
    frame = pd.DataFrame(
        {
            "Pol6TTaCod": ["X", "X", "X", "X", "Y", "Y", "Y", "Y"],
            "CoberturaLabel": ["A", "A", "A", "B", "A", "B", "B", "B"],
            "PrimaRC": [100.0] * 8,
            "PrimaCasco": [20.0] * 8,
            "Prima": [120.0] * 8,
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
