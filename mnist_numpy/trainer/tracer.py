from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from itertools import chain
from pathlib import Path

import h5py
import numpy as np

from mnist_numpy.model.layer import DenseLayer
from mnist_numpy.model.mlp import MultiLayerPerceptron


class TracerStrategy(ABC):
    @abstractmethod
    def should_trace(self, iteration: int) -> bool: ...


class PerEpochTracerStrategy(TracerStrategy):
    def __init__(self, *, training_set_size: int, batch_size: int):
        self._training_set_size = training_set_size
        self._batch_size = batch_size

    def should_trace(self, iteration: int) -> bool:
        return iteration % (self._training_set_size / self._batch_size) == 0


class PerStepTracerStrategy(TracerStrategy):
    def should_trace(self, iteration: int) -> bool:
        return True


class SampleTracerStrategy(TracerStrategy):
    def __init__(self, *, sample_rate: float):
        if sample_rate < 0 or sample_rate > 1:
            raise ValueError("Sample rate must be between 0 and 1")
        self._sample_rate = sample_rate

    def should_trace(self, iteration: int) -> bool:
        del iteration  # unused
        return np.random.rand() < self._sample_rate


@dataclass(kw_only=True, frozen=True)
class TracerConfig:
    trace_activations: bool = True
    trace_biases: bool = True
    trace_raw_gradients: bool = True
    trace_updates: bool = True
    trace_strategy: TracerStrategy
    trace_weights: bool = True


class Tracer:
    def __init__(
        self,
        *,
        model: MultiLayerPerceptron,
        training_log_path: Path,
        tracer_config: TracerConfig,
    ):
        self.model = model
        self._dense_layers: Sequence[DenseLayer] = (
            tuple(  # TODO: Consider reducing coupling.
                chain(
                    (
                        layer
                        for layer in model.hidden_layers
                        if isinstance(layer, DenseLayer)
                    ),
                    (model.output_layer,),  # type: ignore[arg-type]
                ),
            )
        )
        self.trace_logging_path = training_log_path.with_name(
            training_log_path.name.replace("training_log.csv", "trace_log.hdf5")
        )
        self._tracer_config = tracer_config
        self._iterations = 0

        with h5py.File(self.trace_logging_path, "w") as f:
            f.create_group("weights")
            f.create_group("biases")
            f.create_group("raw_gradients")
            f.create_group("updates")
            f.attrs["layer_count"] = len(self._dense_layers)
            f.attrs["iterations"] = 0

    def __call__(
        self,
        gradient: MultiLayerPerceptron.Gradient,
        update: MultiLayerPerceptron.Gradient,
    ) -> None:
        activations = self.model._As[1:]  # Ignore input layer activations.
        self._iterations += 1
        if not self._tracer_config.trace_strategy.should_trace(self._iterations):
            return

        with h5py.File(self.trace_logging_path, "a") as f:
            f.attrs["iterations"] = self._iterations

            iter_group = f.create_group(f"iteration_{self._iterations}")
            iter_group.attrs["timestamp"] = datetime.now().isoformat()

            if self._tracer_config.trace_activations:
                activation_group = iter_group.create_group("activations")
                for i, activation in enumerate(activations):
                    layer_group = activation_group.create_group(f"layer_{i}")
                    hist_values, hist_bins = np.histogram(activation, bins=100)
                    layer_group.create_dataset("histogram_values", data=hist_values)
                    layer_group.create_dataset("histogram_bins", data=hist_bins)

                    layer_group.create_dataset(
                        "deciles", data=np.quantile(activation, np.linspace(0, 1, 11))
                    )

                    layer_group.attrs["mean"] = np.mean(activation)
                    layer_group.attrs["std"] = np.std(activation)
                    layer_group.attrs["min"] = np.min(activation)
                    layer_group.attrs["max"] = np.max(activation)

            if self._tracer_config.trace_weights:
                dense_layer_params = tuple(
                    layer.parameters for layer in self._dense_layers
                )
                weight_group = iter_group.create_group("weights")

                for i, param in enumerate(dense_layer_params):
                    layer_group = weight_group.create_group(f"layer_{i}")
                    hist_values, hist_bins = np.histogram(param._W, bins=100)
                    layer_group.create_dataset("histogram_values", data=hist_values)
                    layer_group.create_dataset("histogram_bins", data=hist_bins)

                    layer_group.create_dataset(
                        "deciles", data=np.quantile(param._W, np.linspace(0, 1, 11))
                    )

                    layer_group.attrs["mean"] = np.mean(param._W)
                    layer_group.attrs["std"] = np.std(param._W)
                    layer_group.attrs["min"] = np.min(param._W)
                    layer_group.attrs["max"] = np.max(param._W)

            if self._tracer_config.trace_biases:
                dense_layer_params = tuple(
                    layer.parameters for layer in self._dense_layers
                )
                bias_group = iter_group.create_group("biases")

                for i, param in enumerate(dense_layer_params):
                    layer_group = bias_group.create_group(f"layer_{i}")
                    hist_values, hist_bins = np.histogram(param._B, bins=100)
                    layer_group.create_dataset("histogram_values", data=hist_values)
                    layer_group.create_dataset("histogram_bins", data=hist_bins)

                    layer_group.create_dataset(
                        "deciles", data=np.quantile(param._B, np.linspace(0, 1, 11))
                    )

                    layer_group.attrs["mean"] = np.mean(param._B)
                    layer_group.attrs["std"] = np.std(param._B)
                    layer_group.attrs["min"] = np.min(param._B)
                    layer_group.attrs["max"] = np.max(param._B)

            if self._tracer_config.trace_raw_gradients:
                dense_layer_gradients = tuple(
                    gradient.dParams[i] for i in range(len(self._dense_layers))
                )
                raw_gradient_group = iter_group.create_group("raw_gradients")

                for i, grad in enumerate(dense_layer_gradients):
                    layer_group = raw_gradient_group.create_group(f"layer_{i}")

                    weight_group = layer_group.create_group("weights")
                    hist_values, hist_bins = np.histogram(grad._W, bins=100)
                    weight_group.create_dataset("histogram_values", data=hist_values)
                    weight_group.create_dataset("histogram_bins", data=hist_bins)

                    weight_group.create_dataset(
                        "deciles", data=np.quantile(grad._W, np.linspace(0, 1, 11))
                    )

                    weight_group.attrs["mean"] = np.mean(grad._W)
                    weight_group.attrs["std"] = np.std(grad._W)
                    weight_group.attrs["min"] = np.min(grad._W)
                    weight_group.attrs["max"] = np.max(grad._W)

                    bias_group = layer_group.create_group("biases")
                    hist_values, hist_bins = np.histogram(grad._B, bins=100)
                    bias_group.create_dataset("histogram_values", data=hist_values)
                    bias_group.create_dataset("histogram_bins", data=hist_bins)

                    bias_group.create_dataset(
                        "deciles", data=np.quantile(grad._B, np.linspace(0, 1, 11))
                    )

                    bias_group.attrs["mean"] = np.mean(grad._B)
                    bias_group.attrs["std"] = np.std(grad._B)
                    bias_group.attrs["min"] = np.min(grad._B)
                    bias_group.attrs["max"] = np.max(grad._B)

            if self._tracer_config.trace_updates:
                update_gradients = tuple(
                    update.dParams[i] for i in range(len(self._dense_layers))
                )
                update_group = iter_group.create_group("updates")
                for i, update_gradient in enumerate(update_gradients):
                    layer_group = update_group.create_group(f"layer_{i}")

                    weight_group = layer_group.create_group("weights")
                    hist_values, hist_bins = np.histogram(update_gradient._W, bins=100)
                    weight_group.create_dataset("histogram_values", data=hist_values)
                    weight_group.create_dataset("histogram_bins", data=hist_bins)

                    weight_group.attrs["mean"] = np.mean(update_gradient._W)
                    weight_group.attrs["std"] = np.std(update_gradient._W)
                    weight_group.attrs["min"] = np.min(update_gradient._W)
                    weight_group.attrs["max"] = np.max(update_gradient._W)

                    weight_group.create_dataset(
                        "deciles",
                        data=np.quantile(update_gradient._W, np.linspace(0, 1, 11)),
                    )

                    bias_group = layer_group.create_group("biases")
                    hist_values, hist_bins = np.histogram(update_gradient._B, bins=100)
                    bias_group.create_dataset("histogram_values", data=hist_values)
                    bias_group.create_dataset("histogram_bins", data=hist_bins)

                    bias_group.create_dataset(
                        "deciles",
                        data=np.quantile(update_gradient._B, np.linspace(0, 1, 11)),
                    )

                    bias_group.attrs["mean"] = np.mean(update_gradient._B)
                    bias_group.attrs["std"] = np.std(update_gradient._B)
                    bias_group.attrs["min"] = np.min(update_gradient._B)
                    bias_group.attrs["max"] = np.max(update_gradient._B)
