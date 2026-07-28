import numpy as np
import pandas as pd

from lpsml.dashboards.counterfactual import build_change_analysis, target_options
from lpsml.dashboards.shared import business_table_columns
from lpsml.reporting.metrics import create_scored_dataset


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
            "TariffCoverage_X__A",
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


def test_scored_reporting_dataset_excludes_tariff_pair_encodings() -> None:
    class ConstantModel:
        def predict(self, features):
            return np.full((len(features), 1), 100.0)

    source = pd.DataFrame(
        {
            "NroPoliza": ["P1", "P2"],
            "Pol6TTaCod": ["X", "Y"],
            "CoberturaLabel": ["A", "B"],
            "TariffCoverage_X__A": [1, 0],
            "Feature": [10.0, 20.0],
            "PrimaRC": [100.0, 100.0],
            "Prima": [100.0, 100.0],
        }
    )
    features = source[["TariffCoverage_X__A", "Feature"]]
    targets = source[["PrimaRC"]]

    scored = create_scored_dataset(
        source,
        features,
        targets,
        total_target_column="Prima",
        model=ConstantModel(),
        train_indices=pd.Index([0]),
        test_indices=pd.Index([1]),
        identity_column="NroPoliza",
    )

    assert "TariffCoverage_X__A" not in scored.columns
    assert {"Pol6TTaCod", "CoberturaLabel"} <= set(scored.columns)


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
