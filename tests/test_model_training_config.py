from math import prod

from sklearn.base import clone

from lpsml.modeling.training import build_pipeline, load_json_config


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
