import functools
import re
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Callable, ParamSpec, TypeVar

import click
import numpy as np
import pandas as pd
from loguru import logger
from matplotlib import pyplot as plt
from more_itertools import sample

from mnist_numpy.data import DATA_DIR, DEFAULT_DATA_PATH, load_data
from mnist_numpy.model import (
    DEFAULT_LEARNING_RATE,
    DEFAULT_MOMENTUM_PARAMETER,
    DEFAULT_NUM_ITERATIONS,
    DEFAULT_TRAINING_LOG_MIN_INTERVAL_SECONDS,
    ModelBase,
    MultilayerPerceptron,
    load_model,
)

P = ParamSpec("P")
R = TypeVar("R")


def training_options(f: Callable[P, R]) -> Callable[P, R]:
    @click.option(
        "-a",
        "--learning-rate",
        help="Set the learning rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
    )
    @click.option(
        "-l",
        "--training-log-path",
        type=Path,
        help="Set the path to the training log file",
        default=None,
    )
    @click.option(
        "-n",
        "--num-iterations",
        help="Set number of iterations",
        type=int,
        default=DEFAULT_NUM_ITERATIONS,
    )
    @click.option(
        "-b",
        "--batch-size",
        type=int,
        help="Set the batch size",
        default=None,
    )
    @click.option(
        "-d",
        "--data-path",
        type=Path,
        help="Set the path to the data file",
        default=DEFAULT_DATA_PATH,
    )
    @click.option(
        "-m",
        "--momentum-parameter",
        type=float,
        help="Set the momentum parameter",
        default=DEFAULT_MOMENTUM_PARAMETER,
    )
    @click.option(
        "-p",
        "--training-log-min-interval-seconds",
        type=int,
        help="Set the minimum interval for logging training progress",
        default=DEFAULT_TRAINING_LOG_MIN_INTERVAL_SECONDS,
    )
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)

    return wrapper


@click.group()
def cli(): ...


@cli.command(help="Train the model")
@training_options
@click.option(
    "-t",
    "--model-type",
    type=click.Choice([MultilayerPerceptron.Serialized._tag]),
    help="The type of model to train",
    default=MultilayerPerceptron.Serialized._tag,
)
@click.option(
    "-i",
    "--dims",
    type=int,
    multiple=True,
    default=(10, 10),
)
def train(
    *,
    batch_size: int | None,
    data_path: Path,
    dims: Sequence[int],
    learning_rate: float,
    model_type: str,
    momentum_parameter: float,
    num_iterations: int,
    training_log_min_interval_seconds: int,
    training_log_path: Path | None,
) -> None:
    X_train, Y_train, X_test, Y_test = load_data(data_path)

    seed = int(time.time())
    np.random.seed(seed)
    logger.info(f"Training model with {seed=}.")

    model: ModelBase
    match model_type:
        case MultilayerPerceptron.Serialized._tag:
            model = MultilayerPerceptron.initialize(X_train.shape[1], *dims, 10)
        case _:
            raise ValueError(f"Invalid model type: {model_type}")

    if training_log_path is None:
        model_path = DATA_DIR / f"{seed}_{model.get_name()}_model_{num_iterations=}.pkl"
        training_log_path = model_path.with_name(f"{model_path.stem}_training_log.csv")
    else:
        model_path = training_log_path.with_name(
            f"{training_log_path.stem.replace('_training_log', '')}.pkl"
        )

    model.train(
        learning_rate=learning_rate,
        num_iterations=num_iterations,
        total_iterations=num_iterations,
        X_train=X_train,
        Y_train=Y_train,
        X_test=X_test,
        Y_test=Y_test,
        training_log_path=training_log_path,
        training_log_min_interval_seconds=training_log_min_interval_seconds,
        batch_size=batch_size,
        momentum_parameter=momentum_parameter,
    ).rename(model_path)
    logger.info(f"Saved output to {model_path}.")


@cli.command(help="Resume training the model")
@training_options
def resume(
    *,
    batch_size: int | None,
    data_path: Path,
    learning_rate: float,
    model_path: Path,
    momentum_parameter: float,
    num_iterations: int | None,
    training_log_min_interval_seconds: int,
    training_log_path: Path,
):
    X_train, Y_train, X_test, Y_test = load_data(data_path)

    if num_iterations is None:
        if (mo := re.search(r"num_iterations=(\d+)", training_log_path.name)) is None:
            raise ValueError(f"Invalid training log path: {training_log_path}")
        total_iterations = int(mo.group(1))
        num_iterations = total_iterations
    else:
        total_iterations = num_iterations

    if not (training_log := pd.read_csv(training_log_path)).empty:
        training_log = training_log.iloc[: np.argmin(training_log.iloc[:, 1]), :]
        num_iterations = total_iterations - int(training_log.iloc[-1, 0])  # type: ignore[arg-type]
        training_log.to_csv(training_log_path, index=False)

    output_path = training_log_path.with_name(
        training_log_path.name.replace("training_log.csv", ".pkl")
    )

    load_model(model_path).train(
        X_test=X_test,
        X_train=X_train,
        Y_test=Y_test,
        Y_train=Y_train,
        batch_size=batch_size,
        learning_rate=learning_rate,
        momentum_parameter=momentum_parameter,
        num_iterations=num_iterations,
        total_iterations=total_iterations,
        training_log_min_interval_seconds=training_log_min_interval_seconds,
        training_log_path=training_log_path,
    ).rename(output_path)
    logger.info(f"Saved output to {output_path}.")


@cli.command(help="Run inference using the model")
@click.option(
    "-m",
    "--model-path",
    help="Set the path to the model file",
    type=Path,
    required=True,
)
@click.option(
    "-d",
    "--data-path",
    type=Path,
    help="Set the path to the data file",
    default=DEFAULT_DATA_PATH,
)
def infer(*, model_path: Path, data_path: Path):
    X_train, Y_train, X_test, Y_test = load_data(data_path)
    model = load_model(model_path)

    Y_pred = model.predict(X_train)
    Y_true = np.argmax(Y_train, axis=1)
    logger.info(f"Training Set Accuracy: {np.sum(Y_pred == Y_true) / len(Y_pred)}")

    Y_pred = model.predict(X_test)
    Y_true = np.argmax(Y_test, axis=1)
    logger.info(f"Test Set Accuracy: {np.sum(Y_pred == Y_true) / len(Y_pred)}")

    plt.figure(figsize=(15, 8))
    plt.suptitle("Mislabelled Samples", fontsize=16)

    sample_indices = sample(np.where(Y_true != Y_pred)[0], 25)
    for idx, i in enumerate(sample_indices):
        plt.subplot(8, 5, idx + 1)
        plt.imshow(X_test[i].reshape(28, 28), cmap="gray")
        plt.title(f"Pred: {Y_pred[i]}, True: {Y_true[i]}")
        plt.axis("off")
    plt.subplot(8, 1, (6, 8))
    unique_labels = list(range(10))
    counts_pred = [np.sum(Y_pred == label) for label in unique_labels]
    counts_true = [np.sum(Y_true == label) for label in unique_labels]
    counts_correct = [
        np.sum((Y_true == Y_pred) & (Y_true == label)) for label in unique_labels
    ]

    bar_width = 0.25
    x = np.arange(len(unique_labels))

    plt.bar(x - bar_width, counts_pred, bar_width, label="Predicted")
    plt.bar(x, counts_true, bar_width, label="True")
    plt.bar(x + bar_width, counts_correct, bar_width, label="Correct")

    plt.xticks(x, [str(label) for label in unique_labels])
    plt.xlabel("Digit")
    plt.ylabel("Count")
    plt.title("Predicted vs. True Label Distribution (Sample)")
    plt.legend()

    plt.tight_layout()
    plt.show()
    plt.tight_layout()
    plt.show()


@cli.command(help="Run explainability analysis")
@click.option(
    "-m",
    "--model-path",
    help="Set the path to the model file",
    required=True,
    type=Path,
)
@click.option(
    "-d",
    "--data-path",
    type=Path,
    help="Set the path to the data file",
    default=DEFAULT_DATA_PATH,
)
def explain(*, model_path: Path, data_path: Path):
    X_train = load_data(data_path)[0]
    model = load_model(model_path)
    W = model._W[0].reshape(28, 28, 10)
    avg = np.average(X_train, axis=0).reshape(28, 28)

    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(np.multiply(W[:, :, i], avg), cmap="gray")
        plt.title(str(i))
        plt.axis("off")
    plt.show()


@cli.command(help="Sample input data", name="sample")
@click.option(
    "-d",
    "--data-path",
    type=Path,
    help="Set the path to the data file",
    default=DEFAULT_DATA_PATH,
)
def sample_data(*, data_path: Path):
    X_train = load_data(data_path)[0]
    sample_indices = sample(range(len(X_train)), 25)
    for idx, i in enumerate(sample_indices):
        plt.subplot(5, 5, idx + 1)
        plt.imshow(X_train[i].reshape(28, 28), cmap="gray")
        plt.axis("off")
    plt.show()
