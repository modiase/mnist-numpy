import sys
from pathlib import Path
from typing import Final

import click
import numpy as np
from loguru import logger
from matplotlib import pyplot as plt
from more_itertools import peekable, sample

from mo_net.data import DATA_DIR, SplitConfig, load_data
from mo_net.log import LogLevel, setup_logging
from mo_net.model import Model
from mo_net.resources import MNIST_TEST_URL, MNIST_TRAIN_URL
from mo_net.train.augment import affine_transform

# MNIST-specific constants
N_DIGITS: Final[int] = 10
MNIST_IMAGE_SIZE: Final[int] = 28


def dataset_split_options(f):
    """Decorator to add dataset split options to CLI commands."""

    @click.option(
        "--train-split",
        type=float,
        help="Set the split for the dataset",
        default=0.8,
    )
    @click.option(
        "--train-split-index",
        type=int,
        help="Set the index for the split of the dataset",
        default=0,
    )
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)

    return wrapper


@click.group()
def mnist_cli():
    """MNIST-specific CLI commands."""
    pass


@mnist_cli.command(help="Run inference using the model", name="infer")
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

    model = Model.load(model_path)

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
        plt.imshow(X_test[i].reshape(MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE), cmap="gray")
        plt.title(f"Pred: {Y_test_pred[i]}, True: {Y_test_true[i]}")
        plt.axis("off")
    plt.subplot(8, 1, (6, 8))
    unique_labels = list(range(N_DIGITS))
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


@mnist_cli.command(help="Sample input data", name="sample")
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
            X_train[i] = affine_transform(
                X_train[i], MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE
            )
    X_train = X_train.reshape(-1, MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE)
    for idx, i in enumerate(sample_indices):
        plt.subplot(5, 5, idx + 1)
        plt.imshow(X_train[i], cmap="gray")
        plt.axis("off")
    plt.show()


if __name__ == "__main__":
    mnist_cli()
