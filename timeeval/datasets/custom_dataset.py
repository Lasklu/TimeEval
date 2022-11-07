from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Optional, NamedTuple
import pandas as pd
import numpy as np

from .dataset import AnomalyDetectionDataset, ClassificationDataset, Dataset
from .analyzer import DatasetAnalyzer
from ..data_types import AlgorithmType, TrainingType, InputDimensionality
TRAIN_PATH_KEY = "train_path"
TEST_PATH_KEY = "test_path"
TYPE_KEY = "type"
PERIOD_KEY = "period"


class CDEntry(NamedTuple):
    test_path: Path
    train_path: Optional[Path]
    details: Dataset


class CustomDataset(ABC):
    def __init__(self, name: str, dataset_object: dict, root_path: Path):
        self.dataset_object = dataset_object
        self.name = name
        self.root_path = root_path
        self._validate_dataset()
        self._analyze_dataset()

    @abstractmethod
    def _validate_dataset(self) -> None:
        pass

    @abstractmethod
    def _analyze_dataset(self) -> None:
        pass

    @abstractmethod
    def _training_type(self, train_path: Path) -> str:
        pass

    def _dataset_id(self, name: str, collection_name: str = "custom") -> Tuple[str, str]:
        return collection_name, name

    def _extract_path(self, obj: dict, key: str) -> Path:
        path_string = obj[key]
        path: Path = self.root_path / path_string
        path = path.resolve()
        return path

    def get_path(self, train: bool):
        if train:
            train_path = self.dataset_object.train_path
            if train_path is None:
                raise ValueError(
                    f"Custom dataset {self.name} is unsupervised and has no training time series!")
            else:
                return train_path

        return self.dataset_object.test_path

    @staticmethod
    def get_class(algorithm_type):
        if algorithm_type == 'classification':
            # TODO need to change this to CustomClassificationDataset
            return ClassificationDataset
        else:
            return CustomDetectionDataset


class CustomDetectionDataset(CustomDataset):
    def __init__(self, name: str, dataset_object: dict, root_path: Path) -> None:
        #super(self, dataset_object).__init__()
        super().__init__(name, dataset_object, root_path)

    def _validate_dataset(self) -> None:
        if TEST_PATH_KEY not in self.dataset_object:
            raise ValueError(
                f"The dataset {self.name} misses the required '{TEST_PATH_KEY}' property.")
        elif not self._extract_path(self.dataset_object, TEST_PATH_KEY).exists():
            raise ValueError(
                f"The test file for dataset {self.name} was not found (property '{TEST_PATH_KEY}')!")
        if TRAIN_PATH_KEY in self.dataset_object and not self._extract_path(self.dataset_object, TRAIN_PATH_KEY).exists():
            raise ValueError(
                f"The train file for dataset {self.name} was not found (property '{TRAIN_PATH_KEY}')!")

    def _analyze_dataset(self) -> CDEntry:
        dataset_id = self._dataset_id(self.name)
        dataset_type = self.dataset_object.get(TYPE_KEY, "unknown")
        period = self.dataset_object.get(PERIOD_KEY, None)

        test_path = self._extract_path(self.dataset_object, TEST_PATH_KEY)
        train_path = None
        if TRAIN_PATH_KEY in self.dataset_object:
            train_path = self._extract_path(
                self.dataset_object, TRAIN_PATH_KEY)

        # get training type by inspecting training file
        training_type = self._training_type(train_path)

        # analyze test time series
        dm = DatasetAnalyzer(dataset_id, is_train=False, algorithm_type=AlgorithmType.ANOMALY_DETECTION,
                             dataset_path=test_path)
        dataset = Dataset.get_class(dm.metadata.algorithm_type)
        self.dataset = dataset(
            datasetId=dataset_id,
            dataset_type=dataset_type,
            training_type=training_type,
            algorithm_type=dm.metadata.algorithm_type,
            dimensions=dm.metadata.dimensions,
            length=dm.metadata.length,
            contamination=dm.metadata.contamination,
            min_anomaly_length=dm.metadata.anomaly_length.min,
            median_anomaly_length=dm.metadata.anomaly_length.median,
            max_anomaly_length=dm.metadata.anomaly_length.max,
            num_anomalies=dm.metadata.num_anomalies,
            period_size=period
        )
        self.test_path = test_path
        self.train_path = train_path
        # return CDEntry(test_path, train_path, )

    def _training_type(self, train_path: Optional[Path]) -> TrainingType:
        if train_path is None:
            return TrainingType.UNSUPERVISED
        else:
            labels = pd.read_csv(train_path).iloc[:, -1]
            if np.any(labels):
                return TrainingType.SUPERVISED
            else:
                return TrainingType.SEMI_SUPERVISED


# class CustomClassificationDataset()
