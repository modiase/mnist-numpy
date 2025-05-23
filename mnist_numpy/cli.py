import atexit
import functools
import os
import re
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Callable, Final, ParamSpec, TypeVar

import click
import numpy as np
from loguru import logger
from matplotlib import pyplot as plt
from more_itertools import peekable, sample

from mnist_numpy.data import (
    DATA_DIR,
    DEFAULT_DATA_PATH,
    OUTPUT_PATH,
    RUN_PATH,
    load_data,
)
from mnist_numpy.functions import (
    LeakyReLU,
    ReLU,
    Tanh,
    parse_activation_fn,
)
from mnist_numpy.model import MultiLayerPerceptron
from mnist_numpy.model.layer.dropout import attach_dropout_layers
from mnist_numpy.protos import ActivationFn, NormalisationType
from mnist_numpy.regulariser.weight_decay import attach_weight_decay_regulariser
from mnist_numpy.train import (
    TrainingParameters,
)
from mnist_numpy.train.exceptions import AbortTraining
from mnist_numpy.train.trainer.parallel import ParallelTrainer
from mnist_numpy.train.trainer.trainer import BasicTrainer, get_optimizer

P = ParamSpec("P")
R = TypeVar("R")

DEFAULT_DIMS: Final[Sequence[int]] = (10, 10)
DEFAULT_LEARNING_RATE_LIMITS: Final[str] = "1e-4, 1e-2"
DEFAULT_NUM_EPOCHS: Final[int] = 1000
MINIMUM_PROGRESS_FOR_SAVING: Final[float] = 0.1
N_DIGITS: Final[int] = 10


def training_options(f: Callable[P, R]) -> Callable[P, R]:
    @click.option(
        "-l",
        "--training-log-path",
        type=Path,
        help="Set the path to the training log file",
        default=None,
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
        "-s",
        "--learning-rate-limits",
        type=lambda x: tuple(float(y) for y in x.split(",")),
        help="Set the learning rate limits",
        default=DEFAULT_LEARNING_RATE_LIMITS,
    )
    @click.option(
        "-o",
        "--optimizer-type",
        type=click.Choice(["adam", "none"]),
        help="The type of optimizer to use",
        default="adam",
    )
    @click.option(
        "--monotonic",
        type=bool,
        is_flag=True,
        help="Use monotonic training",
        default=False,
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
    "-n",
    "--num-epochs",
    help="Set number of epochs",
    type=int,
    default=DEFAULT_NUM_EPOCHS,
)
@click.option(
    "-i",
    "--dims",
    type=int,
    multiple=True,
    default=(),
)
@click.option(
    "-m",
    "--model-path",
    help="Set the path to the model file",
    type=Path,
)
@click.option(
    "-t",
    "--normalisation-type",
    type=click.Choice(
        [NormalisationType.LAYER, NormalisationType.BATCH, NormalisationType.NONE]
    ),
    help="Set the normalisation type",
    default=NormalisationType.LAYER,
)
@click.option(
    "-k",
    "--dropout-keep-probs",
    type=float,
    help="Set the dropout keep probabilities.",
    multiple=True,
    default=(),
)
@click.option(
    "-f",
    "--activation-fn",
    type=click.Choice([ReLU.name, Tanh.name, LeakyReLU.name]),
    help="Set the activation function",
    default=ReLU.name,
    callback=parse_activation_fn,
)
@click.option(
    "--tracing-enabled",
    type=bool,
    is_flag=True,
    help="Enable tracing",
    default=False,
)
@click.option(
    "-l",
    "--regulariser-lambda",
    type=float,
    help="Set the regulariser lambda",
    default=0.0,
)
@click.option(
    "-e",
    "--warmup-epochs",
    type=int,
    help="Set the number of warmup epochs",
    default=100,
)
@click.option(
    "-r",
    "--max-restarts",
    type=int,
    help="Set the maximum number of restarts",
    default=None,
)
@click.option(
    "-w",
    "--workers",
    type=int,
    help="Set the number of workers",
    default=0,
)
@click.option(
    "--no-monitoring",
    type=bool,
    is_flag=True,
    help="Disable monitoring",
    default=False,
)
@click.option(
    "--no-transform",
    type=bool,
    is_flag=True,
    help="Disable transform",
    default=False,
)
@click.option(
    "-y",
    "--history-max-len",
    type=int,
    help="Set the maximum length of the history",
    default=100,
)
def train(
    *,
    activation_fn: ActivationFn,
    batch_size: int | None,
    data_path: Path,
    dims: Sequence[int],
    dropout_keep_probs: Sequence[float],
    history_max_len: int,
    learning_rate_limits: tuple[float, float],
    model_path: Path | None,
    max_restarts: int | None,
    monotonic: bool,
    no_monitoring: bool,
    no_transform: bool,
    normalisation_type: NormalisationType,
    num_epochs: int,
    optimizer_type: str,
    regulariser_lambda: float,
    training_log_path: Path | None,
    tracing_enabled: bool,
    warmup_epochs: int,
    workers: int,
) -> None:
    X_train, Y_train, X_test, Y_test = load_data(data_path)

    seed = int(os.getenv("MNIST_SEED", time.time()))
    np.random.seed(seed)
    logger.info(f"Training model with {seed=}.")

    model: MultiLayerPerceptron
    train_set_size = X_train.shape[0]
    batch_size = batch_size if batch_size is not None else train_set_size

    if model_path is None:
        if len(dims) == 0:
            dims = DEFAULT_DIMS
        model = MultiLayerPerceptron.of(  # type: ignore[call-overload]
            block_dimensions=(
                tuple(
                    map(
                        lambda d: (d,),
                        [
                            X_train.shape[1],
                            *dims,
                            N_DIGITS,
                        ],
                    )
                )
            ),
            activation_fn=activation_fn,
            batch_size=batch_size,
            normalisation_type=normalisation_type,
            tracing_enabled=tracing_enabled,
        )
    else:
        if len(dims) != 0:
            raise ValueError(
                "Dims must not be provided when loading a model from a file."
            )
        model = MultiLayerPerceptron.load(open(model_path, "rb"), training=True)

    if dropout_keep_probs:
        attach_dropout_layers(
            model=model,
            keep_probs=dropout_keep_probs,
            training=True,
        )

    if training_log_path is None:
        model_path = OUTPUT_PATH / f"{int(time.time())}_{model.get_name()}_model.pkl"
        training_log_path = RUN_PATH / (f"{model_path.stem}_training_log.csv")
    else:
        if (re.search(r"_training_log\.csv$", training_log_path.name)) is None:
            training_log_path = training_log_path.with_name(
                f"{training_log_path.stem}_training_log.csv"
            )
        model_path = OUTPUT_PATH / training_log_path.name.replace(
            "training_log.csv", ".pkl"
        )
    training_parameters = TrainingParameters(
        batch_size=batch_size,
        dropout_keep_probs=tuple(dropout_keep_probs),
        history_max_len=history_max_len,
        learning_rate_limits=learning_rate_limits,
        log_path=training_log_path,
        monotonic=monotonic,
        no_monitoring=no_monitoring,
        no_transform=no_transform,
        normalisation_type=normalisation_type,
        num_epochs=num_epochs,
        regulariser_lambda=regulariser_lambda,
        trace_logging=tracing_enabled,
        train_set_size=train_set_size,
        warmup_epochs=warmup_epochs,
        workers=workers,
    )
    optimizer = get_optimizer(optimizer_type, model, training_parameters)
    if regulariser_lambda > 0:
        attach_weight_decay_regulariser(
            lambda_=regulariser_lambda,
            batch_size=batch_size,
            optimizer=optimizer,
            model=model,
        )

    def save_model(model_checkpoint_path: Path | None) -> None:
        if model_checkpoint_path is None:
            return
        model_checkpoint_path.rename(model_path)
        logger.info(f"Saved output to {model_path}.")

    restarts = 0
    if training_parameters.trace_logging:
        max_restarts = 0

    # TODO: The control flow here is a bit messy.  Remove AbortTraining and use
    # a return value instead. Augment TrainingResult to support resuming.
    start_epoch: int = 0
    model_checkpoint_path: Path | None = None
    trainer = (ParallelTrainer if training_parameters.workers > 0 else BasicTrainer)(
        X_test=X_test,
        X_train=X_train,
        Y_test=Y_test,
        Y_train=Y_train,
        model=model,
        optimizer=optimizer,
        start_epoch=start_epoch,
        training_parameters=training_parameters,
        disable_shutdown=training_parameters.workers != 0,
    )
    atexit.register(trainer.shutdown)
    while max_restarts is None or restarts <= max_restarts:
        try:
            if restarts > 0:
                if model_checkpoint_path is None:
                    raise ValueError(
                        "Cannot resume training. Model checkpoint path is not set."
                    )
                training_result = trainer.resume(start_epoch, model_checkpoint_path)
            else:
                training_result = trainer.train()
        except AbortTraining as e:
            logger.exception(e)
            model_checkpoint_path = e.model_checkpoint_path
            if e.model_checkpoint_save_epoch is not None:
                start_epoch = e.model_checkpoint_save_epoch
            restarts += 1
        else:
            save_model(training_result.model_checkpoint_path)
            break


@cli.command(help="Run inference using the model")
@click.option(
    "-m",
    "--model-path",
    help="Set the path to the model file",
    type=Path,
)
@click.option(
    "-d",
    "--data-path",
    type=Path,
    help="Set the path to the data file",
    default=DEFAULT_DATA_PATH,
)
def infer(*, model_path: Path | None, data_path: Path):
    X_train, Y_train, X_test, Y_test = load_data(data_path)

    if model_path is None:
        output_dir = DATA_DIR / "output"
        output_paths = peekable(output_dir.glob("*.pkl"))
        if output_paths.peek() is None:
            logger.error(
                "No model file found in the output directory and no model path provided."
            )
            sys.exit(1)
        model_path = max(output_paths, key=lambda p: p.stat().st_mtime)
        logger.info(f"Using latest model file: {model_path}")
    if not model_path.exists():
        logger.error(f"File not found: {model_path}")
        sys.exit(1)

    model = MultiLayerPerceptron.load(open(model_path, "rb"))

    Y_pred = model.predict(X_train)
    Y_true = np.argmax(Y_train, axis=1)
    logger.info(f"Training Set Accuracy: {np.sum(Y_pred == Y_true) / len(Y_pred)}")

    Y_pred = model.predict(X_test)
    Y_true = np.argmax(Y_test, axis=1)
    logger.info(f"Test Set Accuracy: {np.sum(Y_pred == Y_true) / len(Y_pred)}")

    plt.figure(figsize=(15, 8))
    plt.suptitle("Mislabelled Examples (Sample)", fontsize=16)

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
