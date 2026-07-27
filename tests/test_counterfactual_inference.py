import numpy as np
import pandas as pd

from counterfactual_inference import (
    apply_value_changes,
    build_counterfactual_frame,
    build_selection_mask,
    model_feature_names,
)


def test_build_selection_mask_combines_range_and_category() -> None:
    frame = pd.DataFrame(
        {
            "antig": [1, 2, 4, 6],
            "Pol6TTaCod": ["X", "X", "Y", "X"],
        }
    )
    expression = {
        "all": [
            {"field": "antig", "op": "between", "lower": 2, "upper": 5},
            {"field": "Pol6TTaCod", "op": "eq", "value": "X"},
        ]
    }

    assert build_selection_mask(frame, expression).tolist() == [False, True, False, False]


def test_apply_value_changes_supports_input_and_output_specs() -> None:
    frame = pd.DataFrame({"TasaCasco": [10.0], "PrimaRC": [100.0]})

    changed_input = apply_value_changes(
        frame,
        [{"field": "TasaCasco", "op": "increase_pct", "value": 15}],
        "field",
    )
    changed_output = apply_value_changes(
        frame,
        [{"component": "PrimaRC", "op": "increase_pct", "value": 10}],
        "component",
    )

    assert np.isclose(changed_input.loc[0, "TasaCasco"], 11.5)
    assert np.isclose(changed_output.loc[0, "PrimaRC"], 110.0)


def test_build_counterfactual_frame_applies_both_change_stages() -> None:
    class DummyModel:
        feature_names_in_ = np.array(["TasaCasco", "FactorCasco"])

        def predict(self, features: pd.DataFrame) -> np.ndarray:
            return np.column_stack(
                [
                    features["FactorCasco"] * 100.0,
                    features["TasaCasco"] * 10.0,
                ]
            )

    source = pd.DataFrame(
        {
            "NroPoliza": [1, 2],
            "antig": [3, 8],
            "TasaCasco": [10.0, 20.0],
            "FactorCasco": [1.0, 2.0],
            "PrimaRC": [90.0, 180.0],
            "PrimaCasco": [95.0, 190.0],
            "Prima": [185.0, 370.0],
        }
    )
    scenario = {
        "selection": {
            "all": [
                {"field": "antig", "op": "between", "lower": 2, "upper": 5},
            ]
        },
        "feature_changes": [
            {"field": "TasaCasco", "op": "increase_pct", "value": 15},
        ],
        "prediction_adjustments": [
            {"component": "PrimaRC", "op": "increase_pct", "value": 10},
        ],
    }

    result = build_counterfactual_frame(
        model=DummyModel(),
        source=source,
        scenario=scenario,
        component_names=["PrimaRC", "PrimaCasco"],
        total_name="Prima",
        excluded_columns=["NroPoliza", "PrimaRC", "PrimaCasco", "Prima"],
    )

    assert result["NroPoliza"].tolist() == [1]
    assert np.isclose(result.loc[0, "TasaCasco"], 10.0)
    assert np.isclose(result.loc[0, "TasaCasco_Counterfactual"], 11.5)
    assert np.isclose(result.loc[0, "PrimaRC_Baseline"], 100.0)
    assert np.isclose(result.loc[0, "PrimaCasco_Baseline"], 100.0)
    assert np.isclose(result.loc[0, "Prima_Baseline"], 200.0)
    assert np.isclose(result.loc[0, "PrimaRC_Counterfactual"], 110.0)
    assert np.isclose(result.loc[0, "PrimaCasco_Counterfactual"], 115.0)
    assert np.isclose(result.loc[0, "Prima_Counterfactual"], 225.0)


def test_model_feature_names_supports_older_pipelines() -> None:
    class FittedModel:
        feature_names_in_ = np.array(["feature_a", "feature_b"])

    class Pipeline:
        named_steps = {"model": FittedModel()}

    assert model_feature_names(
        Pipeline(),
        source_columns=["feature_a", "feature_b", "new_feature"],
        excluded_columns=[],
    ) == ["feature_a", "feature_b"]
