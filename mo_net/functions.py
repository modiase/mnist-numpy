import math
from typing import TypeVar, cast

import click
import numpy as np

from mo_net.protos import ActivationFn, ActivationFnName


def cross_entropy(Y_pred: np.ndarray, Y_true: np.ndarray) -> float:
    return -np.sum(Y_true * np.log(np.clip(Y_pred, 1e-15, 1 - 1e-15)))


_X = TypeVar("_X", bound=np.ndarray | float)


class _Softmax:
    name = ActivationFnName("softmax")

    def __call__(self, x: _X) -> _X:
        if isinstance(x, np.ndarray):
            _x = np.atleast_2d(cast(np.ndarray, x))
            return (exp_x := np.exp(_x - np.max(_x, axis=1, keepdims=True))) / np.sum(
                exp_x, axis=1, keepdims=True
            )
        else:
            # TODO: fix-types
            return cast(_X, 1.0)

    def deriv(self, x: _X) -> _X:
        # The derivative of softmax is more complex and depends on the specific context
        # For neural networks with cross-entropy loss, this derivative is often
        # simplified in the backpropagation calculation
        # This implementation is meant to be used with cross-entropy loss
        # where the combined derivative simplifies
        if isinstance(x, np.ndarray):
            # TODO: fix-types
            return cast(_X, np.ones_like(x))
        else:
            # TODO: fix-types
            return cast(_X, 0.0)


softmax = _Softmax()


class _ReLU:
    name = ActivationFnName("relu")

    def __call__(self, x: _X) -> _X:
        if isinstance(x, np.ndarray):
            # TODO: fix-types
            return cast(_X, np.maximum(0, x))
        else:
            # TODO: fix-types
            return cast(_X, max(0, x))

    def deriv(self, x: _X) -> _X:
        if isinstance(x, np.ndarray):
            # TODO: fix-types
            return cast(_X, np.where(x > 0, 1, 0))
        else:
            # TODO: fix-types
            return cast(_X, 1 if x > 0 else 0)


ReLU = _ReLU()


class _LeakyReLU:
    name = ActivationFnName("leaky_relu")

    def __call__(self, x: _X) -> _X:
        if isinstance(x, np.ndarray):
            # TODO: fix-types
            return cast(_X, np.where(x > 0, x, 0.01 * x))
        else:
            # TODO: fix-types
            return cast(_X, max(0.01 * x, 0))

    def deriv(self, x: _X) -> _X:
        if isinstance(x, np.ndarray):
            # TODO: fix-types
            return cast(_X, np.where(x > 0, 1, 0.01))
        else:
            # TODO: fix-types
            return cast(_X, 1 if x > 0 else 0.01)


LeakyReLU = _LeakyReLU()


class _Tanh:
    name = ActivationFnName("tanh")

    def __call__(self, x: _X) -> _X:
        if isinstance(x, np.ndarray):
            # TODO: fix-types
            return cast(_X, np.tanh(x))
        else:
            # TODO: fix-types
            return cast(_X, math.tanh(x))

    def deriv(self, x: _X) -> _X:
        if isinstance(x, np.ndarray):
            # TODO: fix-types
            return cast(_X, 1 - np.tanh(x) ** 2)
        else:
            # TODO: fix-types
            return cast(_X, 1 - math.tanh(x) ** 2)


Tanh = _Tanh()


class _Identity:
    name = ActivationFnName("identity")

    def __call__(self, x: _X) -> _X:
        return x

    def deriv(self, x: _X) -> _X:
        if isinstance(x, np.ndarray):
            # TODO: fix-types
            return cast(_X, np.ones_like(x))
        else:
            # TODO: fix-types
            return cast(_X, 1.0)


Identity = _Identity()


def _get_activation_fn(name: str) -> ActivationFn:
    if name == ReLU.name:
        return ReLU
    elif name == Tanh.name:
        return Tanh
    elif name == LeakyReLU.name:
        return LeakyReLU
    elif name == Identity.name:
        return Identity
    else:
        raise ValueError(f"Unknown activation function: {name}")


def get_activation_fn(name: ActivationFnName) -> ActivationFn:
    return _get_activation_fn(name)


def parse_activation_fn(
    ctx: click.Context, param: click.Parameter, value: object
) -> ActivationFn:
    del ctx, param  # unused
    if value is None:
        raise click.BadParameter("Activation function name is required")
    if not isinstance(value, str):
        raise click.BadParameter("Activation function name must be a string")
    try:
        return _get_activation_fn(value)
    except ValueError as e:
        raise click.BadParameter(str(e)) from e
