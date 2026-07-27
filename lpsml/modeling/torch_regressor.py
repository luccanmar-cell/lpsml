from __future__ import annotations

"""A compact sklearn-compatible regressor for nonnegative premium components."""

from copy import deepcopy
from typing import Any

import numpy as np
import sys
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from torch import nn
from torch.nn import functional as F


# Allow models saved before the package reorganization to remain loadable.
sys.modules.setdefault("torch_regressor", sys.modules[__name__])


class _PremiumNetwork(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_units: int,
        hidden_layers: int,
        target_scale: np.ndarray,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = input_size
        for _ in range(hidden_layers):
            layers.extend((nn.Linear(width, hidden_units), nn.ReLU()))
            width = hidden_units
        layers.append(nn.Linear(width, output_size))
        self.layers = nn.Sequential(*layers)
        self.register_buffer(
            "target_scale",
            torch.as_tensor(target_scale, dtype=torch.float32),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.layers(features)) * self.target_scale


class NonnegativeJointRegressor(RegressorMixin, BaseEstimator):
    """Fit all premium components jointly with nonnegative Softplus outputs."""

    def __init__(
        self,
        hidden_units: int = 128,
        hidden_layers: int = 2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 512,
        max_epochs: int = 200,
        patience: int = 20,
        validation_fraction: float = 0.1,
        final_weight: float = 0.5,
        random_state: int = 42,
        device: str = "cpu",
        num_threads: int = 1,
        verbose: bool = False,
    ) -> None:
        self.hidden_units = hidden_units
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.final_weight = final_weight
        self.random_state = random_state
        self.device = device
        self.num_threads = num_threads
        self.verbose = verbose

    @staticmethod
    def _loss(
        predicted: torch.Tensor,
        actual: torch.Tensor,
        component_scale: torch.Tensor,
        total_scale: torch.Tensor,
        final_weight: float,
    ) -> torch.Tensor:
        component_loss = (((actual - predicted) ** 2).mean(0) / component_scale).mean()
        total_loss = ((actual.sum(1) - predicted.sum(1)) ** 2).mean() / total_scale
        return (1.0 - final_weight) * component_loss + final_weight * total_loss

    def _validate_parameters(self, row_count: int) -> None:
        if self.hidden_units < 1 or self.hidden_layers < 1:
            raise ValueError("hidden_units and hidden_layers must be positive.")
        if self.batch_size < 1 or self.max_epochs < 1 or self.patience < 1:
            raise ValueError("batch_size, max_epochs, and patience must be positive.")
        if not 0 <= self.final_weight <= 1:
            raise ValueError("final_weight must be between 0 and 1.")
        if not 0 < self.validation_fraction < 0.5:
            raise ValueError("validation_fraction must be between 0 and 0.5.")
        if row_count < 3:
            raise ValueError("At least three rows are required.")

    def fit(self, X: Any, y: Any) -> NonnegativeJointRegressor:
        """Fit the network using the joint component and final-premium loss."""
        features, targets = check_X_y(
            X,
            y,
            multi_output=True,
            y_numeric=True,
            dtype=np.float32,
        )
        self._validate_parameters(len(features))
        if targets.ndim == 1:
            targets = targets.reshape(-1, 1)
        if np.any(targets < 0):
            raise ValueError("Premium targets must be nonnegative.")

        self.n_features_in_ = features.shape[1]
        self.n_outputs_ = targets.shape[1]
        torch.set_num_threads(max(1, int(self.num_threads)))
        torch.manual_seed(self.random_state)
        generator = torch.Generator().manual_seed(self.random_state)

        order = torch.randperm(len(features), generator=generator).numpy()
        validation_rows = max(1, int(len(features) * self.validation_fraction))
        validation_index = order[:validation_rows]
        training_index = order[validation_rows:]
        x_train = torch.from_numpy(features[training_index])
        y_train = torch.from_numpy(targets[training_index])
        x_validation = torch.from_numpy(features[validation_index])
        y_validation = torch.from_numpy(targets[validation_index])

        component_scale = y_train.var(0, unbiased=False).clamp_min(1.0)
        total_scale = y_train.sum(1).var(unbiased=False).clamp_min(1.0)
        target_scale = y_train.mean(0).clamp_min(1.0).numpy()
        training_device = torch.device(self.device)
        model = _PremiumNetwork(
            input_size=self.n_features_in_,
            output_size=self.n_outputs_,
            hidden_units=self.hidden_units,
            hidden_layers=self.hidden_layers,
            target_scale=target_scale,
        ).to(training_device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        x_validation = x_validation.to(training_device)
        y_validation = y_validation.to(training_device)
        component_scale = component_scale.to(training_device)
        total_scale = total_scale.to(training_device)
        best_loss = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        stale_epochs = 0

        for epoch in range(self.max_epochs):
            model.train()
            batch_order = torch.randperm(len(x_train), generator=generator)
            for start in range(0, len(x_train), self.batch_size):
                batch = batch_order[start : start + self.batch_size]
                batch_x = x_train[batch].to(training_device)
                batch_y = y_train[batch].to(training_device)
                optimizer.zero_grad()
                loss = self._loss(
                    model(batch_x),
                    batch_y,
                    component_scale,
                    total_scale,
                    self.final_weight,
                )
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                validation_loss = float(
                    self._loss(
                        model(x_validation),
                        y_validation,
                        component_scale,
                        total_scale,
                        self.final_weight,
                    )
                )
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_state = deepcopy(model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.patience:
                    break
            if self.verbose:
                print(f"epoch={epoch + 1} validation_loss={validation_loss:.6f}")

        if best_state is None:
            raise RuntimeError("Training did not produce a valid model state.")
        model.load_state_dict(best_state)
        self.model_ = model.to("cpu").eval()
        self.best_validation_loss_ = best_loss
        self.n_epochs_ = epoch + 1
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict nonnegative premium components."""
        check_is_fitted(self, "model_")
        features = check_array(X, dtype=np.float32)
        if features.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, received {features.shape[1]}."
            )
        with torch.no_grad():
            predictions = self.model_(torch.from_numpy(features)).numpy()
        return predictions
