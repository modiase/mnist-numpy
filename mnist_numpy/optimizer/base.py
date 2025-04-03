from abc import ABC, abstractmethod
from typing import Generic

import numpy as np

from mnist_numpy.model import ModelT


class OptimizerBase(ABC, Generic[ModelT]):
    @abstractmethod
    def update(
        self, model: ModelT, X_train_batch: np.ndarray, Y_train_batch: np.ndarray
    ) -> None: ...

    @abstractmethod
    def report(self) -> str: ...

    @property
    @abstractmethod
    def learning_rate(self) -> float: ...
