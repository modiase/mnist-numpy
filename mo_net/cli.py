import functools
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Callable, Final, Literal, ParamSpec, TypeVar, assert_never

import click
import numpy as np
from loguru import logger
from matplotlib import pyplot as plt
from more_itertools import peekable, sample

from mo_net.data import (
    DATA_DIR,
    DEFAULT_TRAIN_SPLIT,
    OUTPUT_PATH,
    SplitConfig,
    infer_dataset_url,
    load_data,
)
from mo_net.functions import (
    LeakyReLU,
    ReLU,
    Tanh,
    parse_activation_fn,
)
from mo_net.log import LogLevel, setup_logging
from mo_net.model import Model
from mo_net.protos import ActivationFn, NormalisationType
from mo_net.quickstart import mnist_cnn, mnist_mlp
from mo_net.regulariser.weight_decay import attach_weight_decay_regulariser
from mo_net.resources import MNIST_TEST_URL, MNIST_TRAIN_URL
from mo_net.train import (
    TrainingParameters,
)
from mo_net.train.augment import affine_transform
from mo_net.train.backends.log import parse_connection_string
from mo_net.train.run import TrainingRun
from mo_net.train.trainer.parallel import ParallelTrainer
from mo_net.train.trainer.trainer import (
    BasicTrainer,
    OptimizerType,
    TrainingFailed,
    TrainingResult,
    TrainingSuccessful,
    get_optimizer,
)

P = ParamSpec("P")
R = TypeVar("R")


DEFAULT_LEARNING_RATE_LIMITS: Final[str] = "1e-4, 1e-2"
DEFAULT_NUM_EPOCHS: Final[int] = 100
MAX_BATCH_SIZE: Final[int] = 10000
N_DIGITS: Final[int] = 10


def dataset_split_options(f: Callable[P, R]) -> Callable[P, R]:
    @click.option(
        "--train-split",
        type=float,
        help="Set the split for the dataset",
        default=DEFAULT_TRAIN_SPLIT,
    )
    @click.option(
        "--train-split-index",
        type=int,
        help="Set the index for the split of the dataset",
        default=0,
    )
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)

    return wrapper


def training_options(f: Callable[P, R]) -> Callable[P, R]:
    @click.option(
        "-b",
        "--batch-size",
        type=int,
        help="Set the batch size",
        default=None,
    )
    @click.option(
        "-p",
        "--model-output-path",
        type=Path,
        help="Set the path to the model file",
        default=None,
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
    @click.option(
        "-d",
        "--dataset-url",
        type=str,
        help="Set the url to the dataset",
        default=None,
    )
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
            [v.value for v in NormalisationType],
            case_sensitive=False,
        ),
        help="Set the normalisation type",
        default=NormalisationType.LAYER.value,
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
        default=0,
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
    @click.option(
        "-q",
        "--quickstart",
        type=click.Choice(["mnist_mlp", "mnist_cnn"]),
        help="Set the quickstart",
        default=None,
    )
    @click.option(
        "--only-misclassified-examples",
        type=bool,
        is_flag=True,
        help="Only use misclassified examples for training",
        default=False,
    )
    @click.option(
        "--log-level",
        type=click.Choice(tuple(level.lower() for level in LogLevel)),
        help="Set the log level",
        default="info",
        callback=lambda _, __, value: LogLevel(value.upper())
        if isinstance(value, str)
        else LogLevel.INFO,
    )
    @click.option(
        "--quiet",
        type=bool,
        is_flag=True,
        help="Disable logging",
        default=False,
    )
    @click.option(
        "--logging-backend-connection-string",
        type=str,
        help="Set the connection string for the logging backend",
        default=None,
    )
    @dataset_split_options
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)

    return wrapper


def get_model(
    *,
    X_train: np.ndarray,
    activation_fn: ActivationFn,
    batch_size: int,
    dims: Sequence[int],
    dropout_keep_probs: Sequence[float],
    model_path: Path | None,
    normalisation_type: NormalisationType,
    quickstart: Literal["mnist_mlp", "mnist_cnn"] | None,
    tracing_enabled: bool,
    training_parameters: TrainingParameters,
) -> Model:
    if model_path is None:
        match quickstart:
            case None:
                if len(dims) == 0:
                    raise ValueError("Dims must be provided when training a new model.")
                return Model.mlp_of(  # type: ignore[call-overload]
                    module_dimensions=(
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
                    dropout_keep_probs=dropout_keep_probs,
                )
            case "mnist_mlp":
                return mnist_mlp(training_parameters)
            case "mnist_cnn":
                return mnist_cnn(training_parameters)
            case never_quickstart:
                assert_never(never_quickstart)
    else:
        if len(dims) != 0:
            raise ValueError(
                "Dims must not be provided when loading a model from a file."
            )
        return Model.load(open(model_path, "rb"), training=True)


@click.group()
def cli(): ...


@cli.command("train", help="Train the model")
@training_options
def cli_train(*args, **kwargs) -> TrainingResult:
    log_level = kwargs.get("log_level", LogLevel.INFO)
    seed = int(os.getenv("MO_NET_SEED", time.time()))
    kwargs["seed"] = seed
    np.random.seed(seed)
    setup_logging(log_level)
    logger.info(f"Training model with {seed=}.")
    return train(*args, **kwargs)


def train(
    *,
    activation_fn: ActivationFn,
    batch_size: int | None,
    dataset_url: str | None,
    dims: Sequence[int],
    dropout_keep_probs: Sequence[float],
    history_max_len: int,
    log_level: LogLevel,
    logging_backend_connection_string: str,
    learning_rate_limits: tuple[float, float],
    model_path: Path | None,
    max_restarts: int,
    monotonic: bool,
    model_output_path: Path | None,
    no_monitoring: bool,
    no_transform: bool,
    normalisation_type: NormalisationType,
    num_epochs: int,
    optimizer_type: OptimizerType,
    only_misclassified_examples: bool,
    quickstart: Literal["mnist_mlp", "mnist_cnn"] | None,
    quiet: bool,
    regulariser_lambda: float,
    seed: int,
    tracing_enabled: bool,
    train_split: float,
    train_split_index: int,
    warmup_epochs: int,
    workers: int,
) -> TrainingResult:
    dataset_url = infer_dataset_url(quickstart) if dataset_url is None else dataset_url
    if dataset_url is None:
        raise ValueError("No dataset URL provided and no quickstart template used.")
    X_train, Y_train, X_val, Y_val = load_data(
        dataset_url, split=SplitConfig.of(train_split, train_split_index)
    )

    train_set_size = X_train.shape[0]
    if batch_size is None:
        batch_size = min(train_set_size, MAX_BATCH_SIZE)
    elif batch_size > MAX_BATCH_SIZE:
        raise ValueError(f"Batch size must be less than {MAX_BATCH_SIZE}.")

    training_parameters = TrainingParameters(
        batch_size=batch_size,
        dropout_keep_probs=tuple(dropout_keep_probs),
        history_max_len=history_max_len,
        learning_rate_limits=learning_rate_limits,
        log_level=log_level.value,
        max_restarts=max_restarts if not tracing_enabled else 0,
        monotonic=monotonic,
        no_monitoring=no_monitoring,
        no_transform=no_transform,
        normalisation_type=normalisation_type,
        num_epochs=num_epochs,
        quiet=quiet,
        regulariser_lambda=regulariser_lambda,
        trace_logging=tracing_enabled,
        train_set_size=train_set_size,
        warmup_epochs=warmup_epochs,
        workers=workers,
    )

    model = get_model(
        model_path=model_path,
        quickstart=quickstart,
        dims=dims,
        X_train=X_train,
        activation_fn=activation_fn,
        batch_size=batch_size,
        normalisation_type=normalisation_type,
        tracing_enabled=tracing_enabled,
        dropout_keep_probs=dropout_keep_probs,
        training_parameters=training_parameters,
    )

    if model_output_path is None:
        model_output_path = OUTPUT_PATH / f"{int(time.time())}_{model.get_name()}.pkl"
    run = TrainingRun(
        seed=seed,
        backend=parse_connection_string(logging_backend_connection_string),
    )
    optimizer = get_optimizer(optimizer_type, model, training_parameters)
    if regulariser_lambda > 0:
        attach_weight_decay_regulariser(
            lambda_=regulariser_lambda,
            batch_size=batch_size,
            optimizer=optimizer,
            model=model,
        )

    if only_misclassified_examples:
        Y_train_pred = model.predict(X_train)
        Y_train_true = np.argmax(Y_train, axis=1)
        misclassified_indices = np.where(Y_train_pred != Y_train_true)[0]
        X_train = X_train[misclassified_indices]
        Y_train = Y_train[misclassified_indices]
        training_parameters.train_set_size = X_train.shape[0]
        training_parameters.batch_size = min(
            training_parameters.batch_size, X_train.shape[0]
        )

    def save_model(model_checkpoint_path: Path | None) -> None:
        if model_checkpoint_path is None:
            return
        model_checkpoint_path.rename(model_output_path)
        logger.info(f"Saved output to {model_output_path}.")

    restarts = 0

    start_epoch: int = 0
    model_checkpoint_path: Path | None = None
    trainer = (ParallelTrainer if training_parameters.workers > 0 else BasicTrainer)(
        X_train=X_train,
        X_val=X_val,
        Y_train=Y_train,
        Y_val=Y_val,
        disable_shutdown=training_parameters.workers != 0,
        model=model,
        optimizer=optimizer,
        run=run,
        start_epoch=start_epoch,
        training_parameters=training_parameters,
    )
    training_result: TrainingResult | None = None
    try:
        while restarts <= training_parameters.max_restarts:
            if restarts > 0:
                if model_checkpoint_path is None:
                    raise ValueError(
                        "Cannot resume training. Model checkpoint path is not set."
                    )
                training_result = trainer.resume(
                    start_epoch=start_epoch,
                    model_checkpoint_path=model_checkpoint_path,
                )
            else:
                training_result = trainer.train()
            match training_result:
                case TrainingSuccessful() as result:
                    break
                case TrainingFailed() as result:
                    logger.error(result.message)
                    model_checkpoint_path = result.model_checkpoint_path
                    if result.model_checkpoint_save_epoch is not None:
                        start_epoch = result.model_checkpoint_save_epoch
                    restarts += 1
                case never_training_result:
                    assert_never(never_training_result)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    else:
        if not isinstance(training_result, (TrainingSuccessful, TrainingFailed)):
            raise RuntimeError(
                "Training result is not a TrainingSuccessful or TrainingFailed."
            )
        return training_result
    finally:
        if training_result is not None:
            save_model(training_result.model_checkpoint_path)
        trainer.shutdown()


@cli.command(help="Run inference using the model")
@click.option(
    "-m",
    "--model-path",
    help="Set the path to the model file",
    type=Path,
)
@click.option(
    "-d",
    "--dataset-url",
    type=str,
    help="Set the url to the dataset",
    default=MNIST_TRAIN_URL,
)
@click.option(
    "--test-dataset-url",
    type=str,
    help="Set the url to the dataset",
    default=MNIST_TEST_URL,
)
@dataset_split_options
def infer(
    *,
    model_path: Path | None,
    dataset_url: str,
    test_dataset_url: str,
    train_split: float,
    train_split_index: int,
):
    setup_logging(LogLevel.INFO)
    X_train, Y_train, _, __ = load_data(
        dataset_url, split=SplitConfig.of(train_split, train_split_index)
    )

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

    model = Model.load(open(model_path, "rb"))

    Y_train_pred = model.predict(X_train)
    Y_train_true = np.argmax(Y_train, axis=1)
    logger.info(
        f"Training Set Accuracy: {np.sum(Y_train_pred == Y_train_true) / len(Y_train_pred)}"
    )

    X_test, Y_test = load_data(test_dataset_url)
    Y_test_pred = model.predict(X_test)
    Y_test_true = np.argmax(Y_test, axis=1)
    logger.info(
        f"Test Set Accuracy: {np.sum(Y_test_pred == Y_test_true) / len(Y_test_pred)}"
    )

    precision = np.sum(Y_test_pred == Y_test_true) / len(Y_test_pred)
    recall = np.sum(Y_test_true == Y_test_pred) / len(Y_test_true)
    f1_score = 2 * precision * recall / (precision + recall)
    logger.info(f"F1 Score: {f1_score}")

    plt.figure(figsize=(15, 8))
    plt.suptitle("Mislabelled Examples (Sample)", fontsize=16)

    sample_indices = sample(np.where(Y_test_true != Y_test_pred)[0], 25)
    for idx, i in enumerate(sample_indices):
        plt.subplot(8, 5, idx + 1)
        plt.imshow(X_test[i].reshape(28, 28), cmap="gray")
        plt.title(f"Pred: {Y_test_pred[i]}, True: {Y_test_true[i]}")
        plt.axis("off")
    plt.subplot(8, 1, (6, 8))
    unique_labels = list(range(10))
    counts_pred = [np.sum(Y_test_pred == label) for label in unique_labels]
    counts_true = [np.sum(Y_test_true == label) for label in unique_labels]
    counts_correct = [
        np.sum((Y_test_true == Y_test_pred) & (Y_test_true == label))
        for label in unique_labels
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
    "--dataset-url",
    type=str,
    help="Set the url to the dataset",
    default=MNIST_TRAIN_URL,
)
@click.option(
    "--with-transformed",
    type=bool,
    is_flag=True,
    help="Sample the transformed data",
    default=False,
)
def sample_data(*, dataset_url: str, with_transformed: bool):
    X_train = load_data(dataset_url)[0]
    sample_indices = sample(range(len(X_train)), 25)
    if with_transformed:
        for i in sample_indices:
            X_train[i] = affine_transform(X_train[i], 28, 28)
    X_train = X_train.reshape(-1, 28, 28)
    for idx, i in enumerate(sample_indices):
        plt.subplot(5, 5, idx + 1)
        plt.imshow(X_train[i], cmap="gray")
        plt.axis("off")
    plt.show()
