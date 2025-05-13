import operator
from functools import reduce

from mnist_numpy.model.layer.base import Hidden
from mnist_numpy.protos import Activations, D, Dimensions


class Reshape(Hidden):
    def _forward_prop(self, *, input_activations: Activations) -> Activations:
        return input_activations.reshape(
            input_activations.shape[0], *self.output_dimensions
        )

    def _backward_prop(self, *, dZ: D[Activations]) -> D[Activations]:
        return dZ.reshape(dZ.shape[0], *self.input_dimensions)


class Flatten(Reshape):
    def __init__(self, *, input_dimensions: Dimensions):
        super().__init__(
            input_dimensions=input_dimensions,
            output_dimensions=(reduce(operator.mul, input_dimensions, 1),),
        )

    def _forward_prop(self, *, input_activations: Activations) -> Activations:
        return input_activations.reshape(input_activations.shape[0], -1)

    def _backward_prop(self, *, dZ: D[Activations]) -> D[Activations]:
        return dZ.reshape(dZ.shape[0], *self.input_dimensions)
