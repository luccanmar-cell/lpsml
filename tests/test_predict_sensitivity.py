import pandas as pd

# These tests cover the small parsing and feature-alignment helpers that make the
# workflow predictable before running it against a real trained model.

from predict_sensitivity import (
    align_features_to_model,
    apply_modifications,
    parse_modification_specs,
)


def test_parse_and_apply_modifications() -> None:
    df = pd.DataFrame({"feature_a": [1.0, 2.0], "feature_b": [3.0, 4.0]})

    specs = parse_modification_specs(["feature_a=10", "1:feature_b=99"])
    modified = apply_modifications(df, specs)

    assert modified.loc[0, "feature_a"] == 10.0
    assert modified.loc[1, "feature_b"] == 99.0
    assert modified.loc[0, "feature_b"] == 3.0


def test_align_features_to_model() -> None:
    df = pd.DataFrame({"feature_a": [1.0, 2.0], "feature_c": [3.0, 4.0]})

    class DummyModel:
        feature_names_in_ = ["feature_a", "feature_b", "feature_c"]

    aligned = align_features_to_model(df, DummyModel())

    assert aligned.columns.tolist() == ["feature_a", "feature_b", "feature_c"]
    assert aligned.loc[0, "feature_b"] == 0.0
