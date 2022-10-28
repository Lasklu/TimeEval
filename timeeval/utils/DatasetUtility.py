from pathlib import Path

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sktime.datasets import load_from_tsfile, load_from_tsfile_to_dataframe


class BaseDatasetUtility(ABC):
    def extract_labels(self, df: pd.DataFrame) -> np.ndarray:
        return df.values[:, -1].astype(np.float64)

    def extract_features(self, df: pd.DataFrame) -> np.ndarray:
        return df.values[:, 1:-1]

    @abstractmethod
    def load_dataset(self, path: Path) -> pd.DataFrame:
        # return pd.read_csv(path, parse_dates=["timestamp"], infer_datetime_format=True)
        pass

    @abstractmethod
    def load_labels_only(self, path: Path) -> np.ndarray:
        pass


class AnomalyDetectionDatasetUtility(BaseDatasetUtility):
    def load_dataset(self, path: Path) -> pd.DataFrame:
        return pd.read_csv(path, parse_dates=["timestamp"], infer_datetime_format=True)

    def load_labels_only(self, path: Path) -> np.ndarray:
        labels: np.ndarray = pd.read_csv(path, usecols=["is_anomaly"])[
            "is_anomaly"].values.astype(np.float64)
        return labels


class ClassificationDatasetUtility(BaseDatasetUtility):
    def load_dataset(self, path: Path) -> pd.DataFrame:
        return load_from_tsfile_to_dataframe(path, return_separate_X_and_y=False)

    def load_labels_only(self, path: Path) -> np.ndarray:
        _, labels = load_from_tsfile(path, return_data_type="numpy2d")
        return labels.astype(np.float64)
