import numpy as np
import pandas as pd

from lpsml.dashboards.counterfactual import build_change_analysis, target_options
from lpsml.dashboards.shared import business_table_columns


def test_business_table_columns_exclude_model_encodings() -> None:
    frame = pd.DataFrame(
        columns=[
            "NroPoliza",
            "antig",
            "TasaCasco",
            "TasaCasco_Counterfactual",
            "Pol6TTaCod",
            "CoberturaLabel",
            "Accesorios_16",
            "Cobertura_A",
            "Pol6TTaCodEncoded",
            "PrimaRC",
            "Prima",
            "PrimaRC Prediction",
            "Prediction",
            "Prima_Baseline",
            "Prima_Counterfactual",
            "Dataset Split",
            "Absolute Percent Error",
        ]
    )

    columns = business_table_columns(frame)

    assert columns == [
        "NroPoliza",
        "antig",
        "TasaCasco",
        "TasaCasco_Counterfactual",
        "Pol6TTaCod",
        "CoberturaLabel",
        "PrimaRC",
        "Prima",
        "PrimaRC Prediction",
        "Prediction",
        "Prima_Baseline",
        "Prima_Counterfactual",
    ]


def test_counterfactual_change_analysis_is_aligned() -> None:
    frame = pd.DataFrame(
        {
            "NroPoliza": [1, 2],
            "Prima_Baseline": [100.0, 0.0],
            "Prima_Counterfactual": [115.0, 5.0],
        }
    )

    analysis = build_change_analysis(
        frame,
        "Prima_Baseline",
        "Prima_Counterfactual",
    )

    assert analysis["_Change"].tolist() == [15.0, 5.0]
    assert np.isclose(analysis.loc[0, "_PercentChange"], 15.0)
    assert np.isnan(analysis.loc[1, "_PercentChange"])


def test_counterfactual_target_options_require_baselines() -> None:
    frame = pd.DataFrame(
        columns=[
            "Prima_Baseline",
            "Prima_Counterfactual",
            "PrimaRC_Baseline",
            "PrimaRC_Counterfactual",
            "PrimaCasco_Counterfactual",
        ]
    )

    assert target_options(frame) == {
        "Final Prima": ("Prima_Baseline", "Prima_Counterfactual"),
        "PrimaRC": ("PrimaRC_Baseline", "PrimaRC_Counterfactual"),
    }
