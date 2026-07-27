import joblib
import numpy as np
import pandas as pd
import pytest

from lpsml.data.processing import split_valid_model_rows
from lpsml.modeling.torch_regressor import NonnegativeJointRegressor
from lpsml.modeling.training import MODEL_TYPES, predict_premium_components


def test_negative_premium_row_is_doubtful() -> None:
    frame = pd.DataFrame(
        {
            "Pol6TTaCod": ["X", "X", "X"],
            "CoberturaLabel": ["A", "A", "A"],
            "PrimaRC": [100.0, 101.0, 101.0],
            "PrimaCasco": [20.0, 19.0, 0.0],
            "PrimaAccesorio": [0.0, 0.0, -1.0],
            "Prima": [120.0, 120.0, 100.0],
        }
    )

    clean, doubtful = split_valid_model_rows(
        frame,
        component_target_columns=["PrimaRC", "PrimaCasco", "PrimaAccesorio"],
        total_target_column="Prima",
        min_pair_count=2,
    )

    assert clean.index.tolist() == [0, 1]
    assert doubtful.index.tolist() == [2]
    assert doubtful.loc[2, "DoubtfulReason"] == "Negative Prima value"


def test_supported_models_are_nonnegative_joint_estimators() -> None:
    assert set(MODEL_TYPES) == {"random_forest", "extra_trees", "pytorch"}


def test_joint_regressor_is_nonnegative_and_serializable(tmp_path) -> None:
    rng = np.random.default_rng(42)
    features = rng.normal(size=(96, 3)).astype(np.float32)
    targets = np.column_stack(
        [
            np.maximum(2.0 * features[:, 0] + 3.0, 0.0),
            np.maximum(features[:, 1] - features[:, 2], 0.0),
        ]
    ).astype(np.float32)
    model = NonnegativeJointRegressor(
        hidden_units=12,
        hidden_layers=1,
        batch_size=24,
        max_epochs=20,
        patience=5,
        random_state=42,
    ).fit(features, targets)

    predictions = model.predict(features)
    assert predictions.shape == targets.shape
    assert np.all(predictions >= 0)

    model_path = tmp_path / "model.joblib"
    joblib.dump(model, model_path)
    restored = joblib.load(model_path)
    np.testing.assert_allclose(restored.predict(features), predictions)


def test_prediction_contract_rejects_negative_components() -> None:
    class InvalidModel:
        def predict(self, features):
            return np.full((len(features), 2), -1.0)

    with pytest.raises(ValueError, match="negative premium component"):
        predict_premium_components(InvalidModel(), pd.DataFrame({"x": [1.0]}))
