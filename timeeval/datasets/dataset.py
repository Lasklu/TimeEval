from dataclasses import dataclass
from typing import Optional
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from pathlib import Path
from sktime.datasets import load_from_tsfile, load_from_tsfile_to_dataframe

from ..data_types import AlgorithmType, TrainingType, InputDimensionality
from .metadata import DatasetId


@dataclass
class Dataset:
    """Dataset information containing basic metadata about the dataset.

    This class is used within TimeEval heuristics to determine the heuristic values based on the dataset properties.
    """
    datasetId: DatasetId
    dataset_type: str
    training_type: TrainingType
    algorithm_type: AlgorithmType
    length: int
    dimensions: int
    contamination: float
    min_anomaly_length: int
    median_anomaly_length: int
    max_anomaly_length: int
    period_size: Optional[int] = None
    num_anomalies: Optional[int] = None

    @property
    def name(self) -> str:
        return self.datasetId[1]

    @property
    def collection_name(self) -> str:
        return self.datasetId[0]

    @property
    def input_dimensionality(self) -> InputDimensionality:
        return InputDimensionality.from_dimensions(self.dimensions)

    @property
    def has_anomalies(self) -> Optional[bool]:
        if self.num_anomalies is None:
            return None
        else:
            return self.num_anomalies > 0

    @abstractmethod
    def get_dataset(self, path: Path) -> pd.DataFrame:
        # return pd.read_csv(path, parse_dates=["timestamp"], infer_datetime_format=True)
        pass

    @abstractmethod
    def get_labels(self, path: Path) -> np.ndarray:
        pass

    @abstractmethod
    def extract_features(self, df: pd.DataFrame) -> np.ndarray:
        pass

    @staticmethod
    def get_class(algorithm_type):
        if algorithm_type == "classification":
            return ClassificationDataset
        elif algorithm_type == "anomaly_detection":
            return AnomalyDetectionDataset
        else:
            raise ValueError(f"Unknown algorithm type {algorithm_type}!")


@dataclass
class AnomalyDetectionDataset(Dataset):
    def get_dataset(self, path: Path) -> pd.DataFrame:
        return pd.read_csv(path, parse_dates=["timestamp"], infer_datetime_format=True)

    def extract_features(self, df: pd.DataFrame) -> np.ndarray:
        # assumption that first line is timestamp
        features: np.ndarray = df.values[:, 1:-1]
        return features

    def get_labels(self, path: Path) -> np.ndarray:
        labels: np.ndarray = pd.read_csv(path, usecols=["is_anomaly"])[
            "is_anomaly"].values.astype(np.float64)
        return labels


@dataclass
class ClassificationDataset(Dataset):
    def get_dataset(self, path: Path) -> pd.DataFrame:
        return load_from_tsfile_to_dataframe(path, return_separate_X_and_y=False)

    def get_labels(self, path: Path) -> np.ndarray:
        _, labels = load_from_tsfile(path, return_data_type="numpy2d")
        return labels.astype(np.float64)
