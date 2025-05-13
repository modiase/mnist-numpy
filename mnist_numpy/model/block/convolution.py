from typing import Literal

from mnist_numpy.functions import ReLU
from mnist_numpy.model.block.base import Hidden
from mnist_numpy.model.layer.activation import Activation
from mnist_numpy.model.layer.convolution import Convolution2D
from mnist_numpy.model.layer.pool import MaxPooling2D
from mnist_numpy.protos import ActivationFn, Dimensions


class Convolution(Hidden):
    def __init__(
        self,
        *,
        input_dimensions: Dimensions,
        n_kernels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        pool_size: int | tuple[int, int] = 2,
        pool_stride: int | tuple[int, int] = 1,
        pool_type: Literal["max"] = "max",
        activation_function: ActivationFn = ReLU,
    ):
        del pool_type  # unused # TODO: Add average pooling
        super().__init__(
            layers=(
                Convolution2D(
                    input_dimensions=input_dimensions,
                    n_kernels=n_kernels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                ),
                MaxPooling2D(
                    input_dimensions=input_dimensions,
                    pool_size=pool_size,
                    stride=pool_stride,
                ),
                Activation(
                    input_dimensions=input_dimensions,
                    activation_function=activation_function,
                ),
            )
        )
