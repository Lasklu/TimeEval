from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
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
        if algorithm_type == "classification" or algorithm_type == AlgorithmType.CLASSIFICATION:
            return ClassificationDataset
        elif algorithm_type == "anomaly_detection" or algorithm_type == AlgorithmType.ANOMALY_DETECTION:
            return AnomalyDetectionDataset
        else:
            raise ValueError(f"Unknown algorithm type {algorithm_type}!")


# @dataclass
class AnomalyDetectionDataset(Dataset):
    def __init__(self, datasetId: DatasetId, dataset_type: str, training_type: TrainingType, **kwargs):
        super().__init__(datasetId, dataset_type,
                         training_type, AlgorithmType.ANOMALY_DETECTION)
        self.dimensions = kwargs["dimensions"]
        self.length = kwargs["length"]
        self.contamination = kwargs["contamination"]
        self.num_anomalies = kwargs["num_anomalies"]
        self.min_anomaly_length = kwargs["min_anomaly_length"]
        self.median_anomaly_length = kwargs["median_anomaly_length"]
        self.max_anomaly_length = kwargs["max_anomaly_length"]
        self.period_size = kwargs["period_size"]

    def get_dataset(self, path: Path) -> pd.DataFrame:
        return pd.read_csv(path, parse_dates=["timestamp"], infer_datetime_format=True)

    def extract_features(self, df: pd.DataFrame) -> np.ndarray:
        # assumption that first line is timestamp
        features: np.ndarray = df.values[:, 1:-1]
        return features

    def get_labels(self, path: Path) -> np.ndarray:
        labels: np.ndarray = pd.read_csv(path, usecols=["is_anomaly"])[
            "is_anomaly"
        ].values.astype(np.float64)
        return labels


class ClassificationDataset(Dataset):
    def __init__(self, datasetId: DatasetId, dataset_type: str, training_type: TrainingType, **kwargs):
        super().__init__(datasetId, dataset_type,
                         training_type, AlgorithmType.CLASSIFICATION)

    def get_dataset(self, path: Path) -> pd.DataFrame:
        return load_from_tsfile_to_dataframe(path, return_separate_X_and_y=False)

    def get_labels(self, path: Path) -> np.ndarray:
        _, labels = load_from_tsfile_to_dataframe(path)
        if labels.dtype.type == np.str_:
            return pd.factorize(labels, sort=True)[0].astype(np.float64)
        return labels.astype(np.float64)
