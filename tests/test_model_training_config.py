from math import prod

import numpy as np
import pandas as pd
from sklearn.base import clone

from lpsml.modeling.training import build_pipeline, load_json_config, search_models


def test_configured_search_values_are_supported_by_each_model() -> None:
    config = load_json_config("configs/model_training.json")

    for model_config in config["models"]:
        for scaler in model_config["scalers"]:
            pipeline = build_pipeline(
                model_config["type"],
                scaler,
                model_config.get("fixed_params"),
            )
            model_parameters = pipeline.named_steps["model"].get_params()
            configured_parameters = {
                **model_config.get("param_grid", {}),
                **model_config.get("optuna_space", {}),
            }
            assert configured_parameters.keys() <= model_parameters.keys()

            for name, values in model_config.get("param_grid", {}).items():
                for value in values:
                    clone(pipeline).set_params(**{f"model__{name}": value})
            for name, specification in model_config.get(
                "optuna_space", {}
            ).items():
                if specification["type"] == "categorical":
                    values = specification["choices"]
                else:
                    values = [specification["low"], specification["high"]]
                for value in values:
                    clone(pipeline).set_params(**{f"model__{name}": value})


def test_grid_search_sizes_remain_practical() -> None:
    config = load_json_config("configs/model_training.json")
    grid_sizes = {
        model["type"]: (
            len(model["scalers"])
            * prod(len(values) for values in model["param_grid"].values())
        )
        for model in config["models"]
    }

    assert grid_sizes == {
        "random_forest": 32,
        "extra_trees": 32,
        "pytorch": 32,
    }


def test_model_search_scores_pair_stratified_out_of_fold_predictions() -> None:
    features = pd.DataFrame({"Feature": np.arange(18, dtype=float)})
    targets = pd.DataFrame(
        {
            "PrimaRC": 100.0 + features["Feature"],
            "PrimaCasco": 20.0 + features["Feature"] / 2.0,
        }
    )
    pairs = pd.Series(["X/A"] * 6 + ["X/B"] * 6 + ["Y/A"] * 6)
    config = {
        "random_state": 42,
        "search": {"method": "grid"},
        "cv": {"n_splits": 3, "shuffle": True, "n_jobs": 1},
        "models": [
            {
                "name": "Tiny extra trees",
                "type": "extra_trees",
                "enabled": True,
                "scalers": ["none"],
                "fixed_params": {
                    "n_estimators": 8,
                    "random_state": 42,
                    "n_jobs": 1,
                },
                "param_grid": {"max_depth": [None]},
            }
        ],
    }

    _, best_result, results = search_models(features, targets, pairs, config)

    assert len(results) == 1
    assert np.isfinite(best_result["cv_worst_pair_mape_percent"])
    assert "cv_balanced_mse" not in best_result
