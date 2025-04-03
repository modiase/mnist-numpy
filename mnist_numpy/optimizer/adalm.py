import math
from collections import deque
from typing import Final, Protocol

import numpy as np

from mnist_numpy.functions import cross_entropy, softmax
from mnist_numpy.model.base import ModelT
from mnist_numpy.model.mlp import MLP_Gradient
from mnist_numpy.optimizer.base import OptimizerBase

MAX_HISTORY_LENGTH: Final[int] = 2


class Parameters(Protocol):
    batch_size: int
    learning_rate: float
    learning_rate_limits: tuple[float, float]
    learning_rate_rescale_factor_per_epoch: float
    momentum_parameter: float


class AdalmOptimizer(OptimizerBase[ModelT]):
    def __init__(
        self,
        *,
        model: ModelT,
        num_epochs: int,
        train_set_size: int,
        parameters: Parameters,
    ):
        self._iterations_per_epoch = train_set_size / parameters.batch_size
        self._learning_rate = parameters.learning_rate
        self._momentum_parameter = parameters.momentum_parameter
        self._min_momentum_parameter = 0.0
        self._max_momentum_parameter = parameters.momentum_parameter
        self._min_learning_rate = parameters.learning_rate_limits[0]
        self._max_learning_rate = parameters.learning_rate_limits[1]
        self._learning_rate_decay_factor = math.exp(
            (math.log(self._min_learning_rate) - math.log(self._max_learning_rate))
            / num_epochs
        )
        self._learning_rate_rescale_factor = math.exp(
            math.log(parameters.learning_rate_rescale_factor_per_epoch)
            / self._iterations_per_epoch
        )
        self._k_batch = 1 / parameters.batch_size
        self._history = deque(
            (model.empty_gradient(),),
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

        update = MLP_Gradient(
            dWs=tuple(
                -(
                    self._learning_rate * (1 - self._momentum_parameter) * dW
                    + self._momentum_parameter * prev_dW
                )
                for prev_dW, dW in zip(prev_update.dWs, gradient.dWs)
            ),
            dbs=tuple(
                -(
                    self._learning_rate * (1 - self._momentum_parameter) * db
                    + self._momentum_parameter * prev_db
                )
                for prev_db, db in zip(prev_update.dbs, gradient.dbs)
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
