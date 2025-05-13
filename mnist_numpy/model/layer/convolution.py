from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypedDict
import numpy as np
from scipy import signal

from mnist_numpy.model.layer.base import Hidden
from mnist_numpy.protos import Activations, D, Dimensions


class Cache(TypedDict):
    output_activations: Activations | None
    dP: D[Activations] | None


@dataclass(frozen=True, kw_only=True)
class Parameters:
    weights: np.ndarray
    biases: np.ndarray

    @classmethod
    def random(
        cls,
        n_kernels: int,
        in_channels: int,
        in_height: int,
        in_width: int,
        out_height: int,
        out_width: int,
    ) -> Parameters:
        return cls(
            weights=np.random.rand(n_kernels, in_channels, in_height, in_width),
            biases=np.random.rand(n_kernels, out_height, out_width),
        )

    @classmethod
    def ones(
        cls,
        n_kernels: int,
        in_channels: int,
        in_height: int,
        in_width: int,
        out_height: int,
        out_width: int,
    ) -> Parameters:
        return cls(
            weights=np.ones((n_kernels, in_channels, in_height, in_width)),
            biases=np.zeros((n_kernels, out_height, out_width)),
        )


type ParametersType = Parameters
type KernelInitFn = Callable[[int, int, int, int, int, int], ParametersType]


class Convolution2D(Hidden):
    Cache = Cache
    Parameters = Parameters

    def __init__(
        self,
        *,
        input_dimensions: Dimensions,
        n_kernels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        kernel_init_fn: KernelInitFn = Parameters.random,
    ):
        if len(input_dimensions) != 3:
            raise ValueError(
                "input_dimensions must be a 3-tuple: (channel, dim_x, dim_y)."
            )
        if not isinstance(kernel_size, (int, tuple)):
            raise ValueError("Invalid kernel_size. Must be an integer or a pair.")
        self._kernel_width, self._kernel_height = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        if not isinstance(stride, (int, tuple)):
            raise ValueError("Invalid stride. Must be an integer or a pair.")
        self._stride_x, self._stride_y = (
            stride if isinstance(stride, tuple) else (stride, stride)
        )
        if not isinstance(padding, (int, tuple)):
            raise ValueError("Invalid padding. Must be an integer or a pair.")
        self._padding_x, self._padding_y = (
            padding if isinstance(padding, tuple) else (padding, padding)
        )
        self._out_width = (
            input_dimensions[1] + 2 * self._padding_x - self._kernel_width
        ) // self._stride_x + 1
        self._out_height = (
            input_dimensions[2] + 2 * self._padding_y - self._kernel_height
        ) // self._stride_y + 1
        in_channels = input_dimensions[0]
        output_dimensions = (
            in_channels * n_kernels,  # Channel
            self._out_height,  # Y
            self._out_width,  # X
        )
        super().__init__(
            input_dimensions=input_dimensions, output_dimensions=output_dimensions
        )
        self._n_kernels = n_kernels
        self._kernel_weights = kernel_init_fn(
            n_kernels,
            in_channels,
            self._kernel_height,
            self._kernel_width,
            self._out_height,
            self._out_width,
        )
        self._cache = self.Cache(output_activations=None, dP=None)

    def _pad_input(self, input_activations: Activations) -> Activations:
        if self._padding_x == 0 and self._padding_y == 0:
            return input_activations
        return np.pad(
            input_activations,
            (
                (0, 0),
                (0, 0),
                (self._padding_x, self._padding_x),
                (self._padding_y, self._padding_y),
            ),
            mode="constant",
            constant_values=(0, 0),
        )

    def _forward_prop(self, *, input_activations: Activations) -> Activations:
        batch_size, in_channels, in_x, in_y = input_activations.shape

        out_x = (in_x + 2 * self._padding_x - self._kernel_width) // self._stride_x + 1
        out_y = (in_y + 2 * self._padding_y - self._kernel_height) // self._stride_y + 1
        output = np.zeros((batch_size, self._n_kernels, out_x, out_y))

        input_padded = self._pad_input(input_activations)

        # TODO: This is a slow implementation. Explore (vectorized) implementation.
        for b in range(batch_size):
            for k in range(self._n_kernels):
                for c in range(in_channels):
                    output[b, k] += (
                        signal.convolve2d(
                            input_padded[b, c],
                            self._kernel_weights.weights[k, c],
                            mode="valid",
                            boundary="fill",
                            fillvalue=0,
                        )[:: self._stride_x, :: self._stride_y]
                        + self._kernel_weights.biases[k]
                    )

        return output

    def _backward_prop(self, *, dZ: D[Activations]) -> D[Activations]:
        return dZ.reshape(dZ.shape[0], *self.input_dimensions)
