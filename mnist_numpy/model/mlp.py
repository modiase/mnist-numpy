import pickle
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import IO, ClassVar, Final, Self, Sequence

import numpy as np
from loguru import logger
from more_itertools import pairwise

from mnist_numpy.functions import ReLU, deriv_ReLU, softmax
from mnist_numpy.model import ModelBase

MAX_HISTORY_LENGTH: Final[int] = 2


class MultilayerPerceptron(ModelBase):
    @dataclass(frozen=True, kw_only=True)
    class Serialized:
        _tag: ClassVar[str] = "mlp_relu"
        W: tuple[np.ndarray, ...]
        b: tuple[np.ndarray, ...]

    @property
    def layers(self) -> tuple[int, ...]:
        return tuple(w.shape[0] for w in self._W)

    def get_name(self) -> str:
        return f"mlp_relu_{'_'.join(str(layer) for layer in self.layers[1:])}"

    @classmethod
    def get_description(cls) -> str:
        return "Multilayer Perceptron with ReLU activation"

    @classmethod
    def initialize(cls, *dims: int) -> Self:
        return cls(
            [np.random.randn(dim_in, dim_out) for dim_in, dim_out in pairwise(dims)],
            [np.random.randn(dim_out) for dim_out in dims[1:]],
        )

    def __init__(
        self,
        W: Sequence[np.ndarray],
        b: Sequence[np.ndarray],
    ):
        self._W = list(W)
        self._b = list(b)
        self._history = deque(
            ((tuple(np.zeros_like(w) for w in W), tuple(np.zeros_like(b) for b in b)),),
            maxlen=MAX_HISTORY_LENGTH,
        )

    def _forward_prop(
        self, X: np.ndarray
    ) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
        Z = [X @ self._W[0] + self._b[0]]
        for w, b in zip(self._W[1:], self._b[1:]):
            Z.append(ReLU(Z[-1]) @ w + b)
        return tuple(Z), tuple(map(ReLU, Z))

    def _backward_prop(
        self,
        X: np.ndarray,
        Y_true: np.ndarray,
        Z: Sequence[np.ndarray],
        A: Sequence[np.ndarray],
    ) -> tuple[Sequence[np.ndarray], Sequence[np.ndarray]]:
        Y_pred = softmax(Z[-1])
        dZ = Y_pred - Y_true
        _A = [X, *A]
        k = 1 / X.shape[0]

        dW = []
        db = []
        for idx in range(len(self._W) - 1, -1, -1):
            dW.append(k * (_A[idx].T @ dZ))
            db.append(k * np.sum(dZ, axis=0))
            if (
                np.isnan(dW[-1]).any()
                or np.isnan(db[-1]).any()
                or np.isnan(dZ[-1]).any()
            ):
                raise ValueError("Invalid gradient. Aborting training.")
            if idx > 0:
                dZ = (dZ @ self._W[idx].T) * deriv_ReLU(Z[idx - 1])

        return tuple(reversed(dW)), tuple(reversed(db))

    def _update_weights(
        self,
        dWs: Sequence[np.ndarray],
        dbs: Sequence[np.ndarray],
        learning_rate: float,
        momentum_parameter: float,
    ) -> None:
        (prev_dWs, prev_dbs) = self._history[-1]

        dWs_update = [
            learning_rate * (1 - momentum_parameter) * dW + momentum_parameter * prev_dW
            for prev_dW, dW in zip(prev_dWs, dWs)
        ]
        dbs_update = [
            learning_rate * (1 - momentum_parameter) * db + momentum_parameter * prev_db
            for prev_db, db in zip(prev_dbs, dbs)
        ]
        for w, dW in zip(self._W, dWs_update):
            w -= dW
        for b, db in zip(self._b, dbs_update):
            b -= db
        self._history.append((tuple(dWs_update), tuple(dbs_update)))

    def _undo_update(self) -> None:
        (dWs, dbs) = self._history.pop()
        for w, dW in zip(self._W, dWs):
            w += dW
        for b, db in zip(self._b, dbs):
            b += db

    def dump(self, io: IO[bytes]) -> None:
        pickle.dump(self.Serialized(W=tuple(self._W), b=tuple(self._b)), io)

    @classmethod
    def load(cls, source: IO[bytes] | Serialized) -> Self:
        if isinstance(source, cls.Serialized):
            return cls(W=source.W, b=source.b)
        data = pickle.load(source)
        if data._tag != cls.Serialized._tag:
            raise ValueError(f"Invalid model type: {data._tag}")
        return cls(W=data.W, b=data.b)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(softmax(self._forward_prop(X)[1][-1]), axis=1)
