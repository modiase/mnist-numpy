import time
from pathlib import Path

import click
from loguru import logger
import numpy as np
from matplotlib import pyplot as plt
from more_itertools import sample
from mnist_numpy.data import DATA_DIR, load_data
from mnist_numpy.model import (
    LinearRegressionModel,
    MultilayerPerceptron,
    DEFAULT_LEARNING_RATE,
    DEFAULT_NUM_ITERATIONS,
    load_model,
)


@click.group()
def cli(): ...


@cli.command(help="Train the model")
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
    "-t",
    "--model-type",
    type=click.Choice(
        [
            LinearRegressionModel.Serialized._tag,
            MultilayerPerceptron.Serialized._tag,
        ]
    ),
    help="The type of model to train",
)
def train(
    *,
    training_log_path: Path | None,
    num_iterations: int,
    learning_rate: float,
    model_type: str,
) -> None:
    X_train, Y_train, X_test, Y_test = load_data()

    seed = int(time.time())
    np.random.seed(seed)
    logger.info(f"Training model with {seed=}.")

    match model_type:
        case LinearRegressionModel.Serialized._tag:
            model = LinearRegressionModel.initialize(X_train.shape[1], Y_train.shape[1])
        case MultilayerPerceptron.Serialized._tag:
            model = MultilayerPerceptron.initialize(X_train.shape[1], 10, 10)
        case _:
            raise ValueError(f"Invalid model type: {model_type}")

    if training_log_path is None:
        model_path = (
            DATA_DIR
            / f"{seed}_{model_type}_model_{num_iterations=}_{learning_rate=}.pkl"
        )
        training_log_path = model_path.with_name(f"{model_path.stem}_training_log.csv")
    else:
        model_path = training_log_path.with_name(
            f"{training_log_path.stem.replace('_training_log', '')}.pkl"
        )

    model.train(
        learning_rate=learning_rate,
        num_iterations=num_iterations,
        X_train=X_train,
        Y_train=Y_train,
        X_test=X_test,
        Y_test=Y_test,
        training_log_path=training_log_path,
    )

    model.dump(open(model_path, "wb"))


@cli.command(help="Run inference using the model")
@click.option(
    "-m",
    "--model-path",
    help="Set the path to the model file",
    type=Path,
    required=True,
)
def infer(model_path: Path):
    X_train, Y_train, X_test, Y_test = load_data()
    model = load_model(model_path)

    Y_pred = model.predict(X_train)
    Y_true = np.argmax(Y_train, axis=1)
    logger.info(f"Training Set Accuracy: {np.sum(Y_pred == Y_true) / len(Y_pred)}")

    Y_pred = model.predict(X_test)
    Y_true = np.argmax(Y_test, axis=1)
    logger.info(f"Test Set Accuracy: {np.sum(Y_pred == Y_true) / len(Y_pred)}")

    plt.figure(figsize=(15, 8))
    sample_indices = sample(range(len(X_test)), 25)
    for idx, i in enumerate(sample_indices):
        plt.subplot(5, 5, idx + 1)
        plt.imshow(X_test[i].reshape(28, 28), cmap="gray")
        plt.title(f"Pred: {Y_pred[i]}, True: {Y_true[i]}")
        plt.axis("off")
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
def explain(*, model_path: Path):
    X_train = load_data()[0]
    model = load_model(model_path)
    W = model._W.reshape(28, 28, 10)
    avg = np.average(X_train, axis=0).reshape(28, 28)

    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(np.multiply(W[:, :, i], avg), cmap="gray")
        plt.title(str(i))
        plt.axis("off")
    plt.show()


@cli.command(help="Sample input data", name="sample")
def sample_():
    X_train = load_data()[0]
    sample_indices = sample(range(len(X_train)), 25)
    for idx, i in enumerate(sample_indices):
        plt.subplot(5, 5, idx + 1)
        plt.imshow(X_train[i].reshape(28, 28), cmap="gray")
        plt.axis("off")
    plt.show()
