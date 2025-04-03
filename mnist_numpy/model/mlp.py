from __future__ import annotations

import pickle
from collections.abc import MutableSequence
from dataclasses import dataclass
from typing import IO, ClassVar, Self, Sequence

import numpy as np
from more_itertools import pairwise

from mnist_numpy.functions import ReLU, deriv_ReLU, softmax
from mnist_numpy.model import ModelBase


@dataclass(frozen=True, kw_only=True)
class MLP_Gradient:
    dW: Sequence[np.ndarray]
    db: Sequence[np.ndarray]

    def __add__(self, other: Self | MLP_Parameters) -> Self:
        match other:
            case MLP_Parameters():
                return self.__class__(
                    dW=tuple(
                        dW1 + dW2 for dW1, dW2 in zip(self.dW, other.W, strict=True)
                    ),
                    db=tuple(
                        db1 + db2 for db1, db2 in zip(self.db, other.b, strict=True)
                    ),
                )
            case MLP_Gradient():
                return self.__class__(
                    dW=tuple(
                        dW1 + dW2 for dW1, dW2 in zip(self.dW, other.dW, strict=True)
                    ),
                    db=tuple(
                        db1 + db2 for db1, db2 in zip(self.db, other.db, strict=True)
                    ),
                )
            case _:
                return NotImplemented

    def __radd__(self, other: MLP_Parameters) -> Self:
        return self + other

    def __neg__(self) -> Self:
        return self.__class__(
            dW=tuple(-_dW for _dW in self.dW),
            db=tuple(-_db for _db in self.db),
        )

    def __sub__(self, other: Self) -> Self:
        return self + (-other)

    def __mul__(self, other: float) -> Self:
        return self.__class__(
            dW=tuple(_dW * other for _dW in self.dW),
            db=tuple(_db * other for _db in self.db),
        )

    def __rmul__(self, other: float) -> Self:
        return self * other

    def __truediv__(self, other: float) -> Self:
        return self * (1 / other)

    def __pow__(self, exp: float) -> Self:
        return self.__class__(
            dW=tuple(_dW**exp for _dW in self.dW),
            db=tuple(_db**exp for _db in self.db),
        )

    def __getitem__(self, idx: int | tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
        match idx:
            case int():
                return self.dW[idx], self.db[idx]
            case tuple():
                i, *rest = idx
                return self.dW[i][*rest], self.db[i][*rest]
            case _:
                return NotImplemented

    @property
    def norm(self) -> tuple[Sequence[float], Sequence[float]]:
        return tuple(np.linalg.norm(_dW) for _dW in self.dW), tuple(
            np.linalg.norm(_db) for _db in self.db
        )


@dataclass(kw_only=True)
class MLP_Parameters:
    W: MutableSequence[np.ndarray]
    b: MutableSequence[np.ndarray]

    @dataclass(frozen=True, kw_only=True)
    class Frozen:
        W: Sequence[np.ndarray]
        b: Sequence[np.ndarray]

        @classmethod
        def from_gradient(cls, gradient: MLP_Gradient) -> Self:
            return cls(
                W=tuple(gradient.dW),
                b=tuple(gradient.db),
            )

        def __add__(self, other: Self) -> Self:
            match other:
                case MLP_Parameters():
                    if len(other.W) != len(self.W) or len(other.b) != len(self.b):
                        raise ValueError("Shape mismatch")
                    for i in range(len(other.W)):
                        other.W[i] += self.W[i]
                    for i in range(len(other.b)):
                        other.b[i] += self.b[i]
                    return other
                case MLP_Parameters.Frozen():
                    return self.__class__(
                        W=tuple(
                            W1 + W2 for W1, W2 in zip(self.W, other.W, strict=True)
                        ),
                        b=tuple(
                            b1 + b2 for b1, b2 in zip(self.b, other.b, strict=True)
                        ),
                    )
                case _:
                    return NotImplemented

        def __radd__(self, other: MLP_Parameters) -> Self:
            return self + other

        def __mul__(self, other: float) -> Self:
            return self.__class__(
                W=tuple(W * other for W in self.W),
                b=tuple(b * other for b in self.b),
            )

        def __rmul__(self, other: float) -> Self:
            return self * other

        def __neg__(self) -> Self:
            return self.__class__(
                W=tuple(-W for W in self.W),
                b=tuple(-b for b in self.b),
            )

        def __sub__(self, other: Self) -> Self:
            return self + (-other)

        def __rsub__(self, other: Self) -> Self:
            return other + (-self)

        def __truediv__(self, other: float) -> Self:
            return self * (1 / other)

        def __rtruediv__(self, other: float) -> Self:
            return self * (1 / other)

        def __getitem__(
            self, idx: int | tuple[int, ...]
        ) -> tuple[np.ndarray, np.ndarray]:
            match idx:
                case int():
                    return self.W[idx], self.b[idx]
                case tuple():
                    i, *rest = idx
                    return self.W[i][*rest], self.b[i][*rest]
                case _:
                    return NotImplemented

        @property
        def norm(self) -> tuple[Sequence[float], Sequence[float]]:
            return tuple(np.linalg.norm(_W) for _W in self.W), tuple(
                np.linalg.norm(_b) for _b in self.b
            )

        def unroll(self) -> tuple[Sequence[np.ndarray], Sequence[np.ndarray]]:
            return tuple(w.flatten() for w in self.W), tuple(b for b in self.b)

    def freeze(self) -> Frozen:
        return self.Frozen(W=tuple(self.W), b=tuple(self.b))


class MultilayerPerceptron(ModelBase[MLP_Parameters, MLP_Gradient]):
    @dataclass(frozen=True, kw_only=True)
    class Serialized:
        _tag: ClassVar[str] = "mlp_relu"
        W: tuple[np.ndarray, ...]
        b: tuple[np.ndarray, ...]

    @property
    def layers(self) -> Sequence[int]:
        return tuple(w.shape[0] for w in self._params.W)

    def get_name(self) -> str:
        return f"mlp_relu_{'_'.join(str(layer) for layer in self.layers[1:])}"

    @classmethod
    def get_description(cls) -> str:
        return "Multilayer Perceptron with ReLU activation"

    @classmethod
    def initialize(cls, *dims: int) -> Self:
        return cls(
            params=MLP_Parameters(
                W=[
                    np.random.randn(dim_in, dim_out)
                    for dim_in, dim_out in pairwise(dims)
                ],
                b=[np.random.randn(dim_out) for dim_out in dims[1:]],
            ),
        )

    def __init__(
        self,
        *,
        params: MLP_Parameters | None = None,
    ):
        self._params = params

    def _forward_prop(
        self, X: np.ndarray
    ) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
        Z = [X @ self._params.W[0] + self._params.b[0]]
        for w, b in zip(self._params.W[1:], self._params.b[1:], strict=True):
            Z.append(ReLU(Z[-1]) @ w + b)
        return tuple(Z), tuple(map(ReLU, Z))

    def _backward_prop(
        self,
        X: np.ndarray,
        Y_true: np.ndarray,
        Z: Sequence[np.ndarray],
        A: Sequence[np.ndarray],
    ) -> MLP_Gradient:
        dZ = softmax(Z[-1]) - Y_true
        _A = [X, *A]
        k = 1 / X.shape[0]

        dW = []
        db = []
        for idx in range(len(self._params.W) - 1, -1, -1):
            dW.append(k * (_A[idx].T @ dZ))
            db.append(k * np.sum(dZ, axis=0))
            if (
                np.isnan(dW[-1]).any()
                or np.isnan(db[-1]).any()
                or np.isnan(dZ[-1]).any()
            ):
                raise ValueError("Invalid gradient. Aborting training.")
            if idx > 0:
                dZ = (dZ @ self._params.W[idx].T) * deriv_ReLU(Z[idx - 1])

        return MLP_Gradient(
            dW=tuple(reversed(dW)),
            db=tuple(reversed(db)),
        )

    def update_parameters(
        self,
        update: MLP_Parameters.Frozen,
    ) -> None:
        self._params += update

    def dump(self, io: IO[bytes]) -> None:
        pickle.dump(
            self.Serialized(W=tuple(self._params.W), b=tuple(self._params.b)), io
        )

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

    def empty_gradient(self) -> MLP_Gradient:
        return MLP_Gradient(
            dW=tuple(np.zeros_like(w) for w in self._params.W),
            db=tuple(np.zeros_like(b) for b in self._params.b),
        )

    def empty_parameters(self) -> MLP_Parameters.Frozen:
        return MLP_Parameters.Frozen(
            W=tuple(np.zeros_like(w) for w in self._params.W),
            b=tuple(np.zeros_like(b) for b in self._params.b),
        )

    @property
    def parameters(self) -> MLP_Parameters.Frozen:
        return self._params.freeze()
