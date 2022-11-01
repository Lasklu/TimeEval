from pathlib import Path

import numpy as np
import pandas as pd

# Todo replace references by dataset class


def load_dataset(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["timestamp"], infer_datetime_format=True)


def load_labels_only(path: Path) -> np.ndarray:
    labels: np.ndarray = pd.read_csv(path, usecols=["is_anomaly"])[
        "is_anomaly"].values.astype(np.float64)
    return labels
