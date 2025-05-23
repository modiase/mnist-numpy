from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

DATA_DIR: Final[Path] = Path(__file__).parent.parent / "data"
DEFAULT_DATA_PATH: Final[Path] = DATA_DIR / "mnist_test.csv"
MAX_PIXEL_VALUE: Final[int] = 255
N_DIGITS: Final[int] = 10
OUTPUT_PATH: Final[Path] = DATA_DIR / "output"
if not OUTPUT_PATH.exists():
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
RUN_PATH: Final[Path] = DATA_DIR / "run"
if not RUN_PATH.exists():
    RUN_PATH.mkdir(parents=True, exist_ok=True)


def load_data(data_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(data_path)
    split_index: int = int(len(df) * 0.8)
    training_set = df.iloc[:split_index, :]
    test_set = df.iloc[split_index:, :]

    Y_train = np.eye(N_DIGITS)[training_set.iloc[:, 0].to_numpy()]
    Y_test = np.eye(N_DIGITS)[test_set.iloc[:, 0].to_numpy()]

    X_train = np.array(training_set.iloc[:, 1:]) / MAX_PIXEL_VALUE
    X_test = np.array(test_set.iloc[:, 1:]) / MAX_PIXEL_VALUE
    return X_train, Y_train, X_test, Y_test
