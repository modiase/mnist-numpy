import math
from collections import deque
from typing import Any, Final, Protocol

import numpy as np

from mnist_numpy.functions import cross_entropy, softmax
from mnist_numpy.model.mlp import (  # TODO: Remove dependence
    MLP_Gradient,
    MLP_Parameters,
)
from mnist_numpy.optimizer.base import ModelT, OptimizerBase

MAX_HISTORY_LENGTH: Final[int] = 2


class Parameters(Protocol):
    batch_size: int
    learning_rate: float
    momentum_parameter: float
    learning_rate_limits: tuple[float, float]
    learning_rate_rescale_factor_per_epoch: float


class NoOptimizer(OptimizerBase[Any]):
    def __init__(
        self,
        *,
        training_parameters: Parameters,
    ):
        self._learning_rate = training_parameters.learning_rate
        self._iterations = 0

    def update(
        self, model: ModelT, X_train_batch: np.ndarray, Y_train_batch: np.ndarray
    ) -> None:
        A_train_batch, Z_train_batch = model._forward_prop(X_train_batch)
        gradient = model._backward_prop(
            X_train_batch, Y_train_batch, Z_train_batch, A_train_batch
        )
        model.update_parameters(
            -self._learning_rate * MLP_Parameters.Frozen.from_gradient(gradient)
        )
        self._iterations += 1

    def report(self) -> str:
        return ""

    @property
    def learning_rate(self) -> float:
        return self._learning_rate


class AdalmOptimizer(OptimizerBase[ModelT]):
    def __init__(
        self,
        *,
        model: ModelT,
        num_epochs: int,
        train_set_size: int,
        training_parameters: Parameters,
    ):
        self._iterations_per_epoch = train_set_size / training_parameters.batch_size
        self._learning_rate = training_parameters.learning_rate
        self._momentum_parameter = training_parameters.momentum_parameter
        self._min_momentum_parameter = 0.0
        self._max_momentum_parameter = training_parameters.momentum_parameter
        self._min_learning_rate = training_parameters.learning_rate_limits[0]
        self._max_learning_rate = training_parameters.learning_rate_limits[1]
        self._learning_rate_decay_factor = math.exp(
            (math.log(self._min_learning_rate) - math.log(self._max_learning_rate))
            / num_epochs
        )
        self._learning_rate_rescale_factor = math.exp(
            math.log(training_parameters.learning_rate_rescale_factor_per_epoch)
            / self._iterations_per_epoch
        )
        self._k_batch = 1 / training_parameters.batch_size
        self._history = deque(
            (model.empty_parameters(),),
            maxlen=MAX_HISTORY_LENGTH,
        )
        self._iterations = 0

    @property
    def learning_rate(self) -> float:
        return self._learning_rate

    def update(
        self,
        model: ModelT,
        X_train_batch: np.ndarray,
        Y_train_batch: np.ndarray,
    ) -> None:
        Z_train_batch, A_train_batch = model._forward_prop(X_train_batch)
        L_batch_before = self._k_batch * cross_entropy(
            softmax(A_train_batch[-1]), Y_train_batch
        )
        # TODO: Refactor further to reduce coupling between model and optimizer.
        gradient = model._backward_prop(
            X_train_batch,
            Y_train_batch,
            Z_train_batch,
            A_train_batch,
        )

        prev_update = self._history[-1]

        update = MLP_Parameters.Frozen(
            W=tuple(
                -self._learning_rate * (1 - self._momentum_parameter) * dW
                + self._momentum_parameter * prev_dW
                for prev_dW, dW in zip(prev_update.W, gradient.dW, strict=True)
            ),
            b=tuple(
                -self._learning_rate * (1 - self._momentum_parameter) * db
                + self._momentum_parameter * prev_db
                for prev_db, db in zip(prev_update.b, gradient.db, strict=True)
            ),
        )
        model.update_parameters(update)
        self._history.append(update)

        _, A_train_batch = model._forward_prop(X_train_batch)
        L_batch_after = self._k_batch * cross_entropy(
            softmax(A_train_batch[-1]), Y_train_batch
        )
        if L_batch_after < L_batch_before:
            self._learning_rate *= 1 + self._learning_rate_rescale_factor
            self._momentum_parameter += 0.05
        else:
            self._learning_rate *= 1 - 2 * self._learning_rate_rescale_factor
            self._momentum_parameter -= 0.05

        self._momentum_parameter = min(
            self._max_momentum_parameter,
            max(
                self._momentum_parameter,
                self._min_momentum_parameter,
            ),
        )
        self._learning_rate = min(
            self._max_learning_rate,
            max(
                self._learning_rate,
                self._min_learning_rate,
            ),
        )
        self._iterations += 1
        if self._iterations % self._iterations_per_epoch == 0:
            self._max_learning_rate *= self._learning_rate_decay_factor

    def report(self) -> str:
        return (
            f"Learning Rate: {self._learning_rate:.10f}, Maximum Learning Rate: {self._max_learning_rate:.10f}"
            f", Momentum Parameter: {self._momentum_parameter:.2f}"
        )


class AdamOptimizer(OptimizerBase[ModelT]):
    def __init__(
        self,
        *,
        model: ModelT,
        learning_rate: float,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        self._beta_1 = beta_1
        self._beta_2 = beta_2
        self._epsilon = epsilon
        self._first_moment = model.empty_gradient()
        self._learning_rate = learning_rate
        self._second_moment = model.empty_gradient()
        self._iterations = 0

    def update(
        self,
        model: ModelT,
        X_train_batch: np.ndarray,
        Y_train_batch: np.ndarray,
    ) -> None:
        self._iterations += 1
        Z_train_batch, A_train_batch = model._forward_prop(X_train_batch)
        gradient = model._backward_prop(
            X_train_batch,
            Y_train_batch,
            Z_train_batch,
            A_train_batch,
        )
        self._first_moment = (
            self._beta_1 * self._first_moment + (1 - self._beta_1) * gradient
        )
        self._second_moment = (
            self._beta_2 * self._second_moment + (1 - self._beta_2) * gradient**2
        )
        first_moment_corrected = self._first_moment / (
            1 - self._beta_1**self._iterations
        )
        second_moment_corrected = self._second_moment / (
            1 - self._beta_2**self._iterations
        )

        update = MLP_Gradient(
            dW=tuple(
                -self._learning_rate
                * first_moment_corrected
                / (np.sqrt(second_moment_corrected) + self._epsilon)
                for first_moment_corrected, second_moment_corrected in zip(
                    first_moment_corrected.dWs, second_moment_corrected.dWs, strict=True
                )
            ),
            db=tuple(
                -self._learning_rate
                * first_moment_corrected
                / (np.sqrt(second_moment_corrected) + self._epsilon)
                for first_moment_corrected, second_moment_corrected in zip(
                    first_moment_corrected.dbs, second_moment_corrected.dbs, strict=True
                )
            ),
        )
        model.update_parameters(update)

    @property
    def learning_rate(self) -> float:
        return self._learning_rate

    def report(self) -> str:
        return f"Learning Rate: {self._learning_rate:.10f}"
