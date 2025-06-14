from math import ceil
from pathlib import Path
from typing import Final, overload

import numpy as np
import pandas as pd
from loguru import logger

from mo_net.resources import MNIST_TRAIN_URL, get_resource

DATA_DIR: Final[Path] = Path(__file__).parent.parent / "data"
MAX_PIXEL_VALUE: Final[int] = 255
N_DIGITS: Final[int] = 10
OUTPUT_PATH: Final[Path] = DATA_DIR / "output"
if not OUTPUT_PATH.exists():
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
RUN_PATH: Final[Path] = DATA_DIR / "run"
if not RUN_PATH.exists():
    RUN_PATH.mkdir(parents=True, exist_ok=True)


def _load_data(data_path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(data_path)
    return (
        np.array(df.iloc[:, 1:]) / MAX_PIXEL_VALUE,
        np.eye(N_DIGITS)[df.iloc[:, 0].to_numpy()],
    )


def _load_data_split(
    data_path: Path, split: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if split <= 0 or split >= 1:
        raise ValueError("Split must be in the interval (0,1)")
    df = pd.read_csv(data_path)
    split_index: int = ceil(len(df) * split)
    training_set = df.iloc[:split_index, :]
    test_set = df.iloc[split_index:, :]

    Y_train = np.eye(N_DIGITS)[training_set.iloc[:, 0].to_numpy()]
    Y_val = np.eye(N_DIGITS)[test_set.iloc[:, 0].to_numpy()]

    X_train = np.array(training_set.iloc[:, 1:]) / MAX_PIXEL_VALUE
    X_val = np.array(test_set.iloc[:, 1:]) / MAX_PIXEL_VALUE
    return X_train, Y_train, X_val, Y_val


@overload
def load_data(
    dataset_url: str, split: None = None
) -> tuple[np.ndarray, np.ndarray]: ...
@overload
def load_data(
    dataset_url: str, split: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...


def load_data(
    dataset_url: str, split: float | None = None
) -> (
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray]
):
    logger.info(f"Loading data from {dataset_url}.")
    data_path = get_resource(dataset_url)
    return (
        _load_data_split(data_path, split)
        if split is not None
        else _load_data(data_path)
    )


def infer_dataset_url(quickstart: str | None) -> str | None:
    if quickstart == "mnist_mlp" or quickstart == "mnist_cnn":
        return MNIST_TRAIN_URL
    return None
