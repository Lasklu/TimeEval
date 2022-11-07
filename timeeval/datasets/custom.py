import json
import warnings
from pathlib import Path
from typing import List, Union, Tuple, Optional, Dict, NamedTuple

import numpy as np
import pandas as pd

from .analyzer import DatasetAnalyzer
from .custom_base import CustomDatasetsBase
from .dataset import AnomalyDetectionDataset, ClassificationDataset, Dataset
from .metadata import DatasetId
from ..data_types import AlgorithmType, TrainingType, InputDimensionality
from .custom_dataset import CustomDataset


class CustomDatasets(CustomDatasetsBase):
    # THIS DOES NOT WORK ANYMORE. NEEDS MORE SOPHISTICATED IMPLEMENTATION
    """Implementation of the custom datasets API.

    Internal API! You should **not need to use or modify** this class.

    This class behaves similar to the :class:`timeeval.datasets.datasets.Datasets`-API while using a different internal
    representation for the dataset index.
    """

    def __init__(self, dataset_config: Union[str, Path]):
        super().__init__()
        dataset_config_path = Path(dataset_config)
        with dataset_config_path.open("r") as f:
            config = json.load(f)

        root_path: Path = dataset_config_path.parent

        store = {}
        for dataset in config:
            if 'algorithm_type' not in config[dataset]:
                raise ValueError(
                    f"The dataset {dataset} misses the required 'algorithm_type' property.")
            algorithmType = config[dataset]['algorithm_type']
            store[dataset] = CustomDataset.get_class(
                algorithmType)(dataset, config[dataset], root_path)

        self._dataset_store: Dict[str, CustomDataset] = store

    def get_collection_names(self) -> List[str]:
        return ["custom"]

    def get_dataset_names(self) -> List[str]:
        return [name for name in self._dataset_store]

    def get(self, dataset_name: str) -> Dataset:
        return self._dataset_store[dataset_name].dataset

    def get_path(self, dataset_name: str, train: bool) -> Path:
        dataset = self._dataset_store[dataset_name]

        if train:
            train_path = dataset.train_path
            if train_path is None:
                raise ValueError(
                    f"Custom dataset {dataset_name} is unsupervised and has no training time series!")
            else:
                return train_path

        return dataset.test_path

    def select(self,
               collection: Optional[str] = None,
               dataset: Optional[str] = None,
               dataset_type: Optional[str] = None,
               algorithm_type: Optional[AlgorithmType] = None,
               datetime_index: Optional[bool] = None,
               training_type: Optional[TrainingType] = None,
               train_is_normal: Optional[bool] = None,
               input_dimensionality: Optional[InputDimensionality] = None,
               min_anomalies: Optional[int] = None,
               max_anomalies: Optional[int] = None,
               max_contamination: Optional[float] = None
               ) -> List[DatasetId]:
        # filter for classification and anomaly detection datasets and then apply selection
        if algorithm_type is None:
            raise ValueError(
                "The algorithm_type parameter is required for custom datasets.")
        if (collection is not None and collection not in self.get_collection_names()) or (
                dataset is not None and dataset not in self.get_dataset_names()):
            return []
        else:
            selectors = []
            # used for an early-skip already
            # if dataset is not None:
            #     selectors.append(lambda meta: meta.datasetId[1] == dataset)
            if dataset_type is not None:
                selectors.append(
                    lambda meta: meta.dataset_type == dataset_type)
            if datetime_index is not None:
                warnings.warn("Filter for index type (datetime or int) is not supported for custom dataset! "
                              "Ignoring it!")
            if training_type is not None:
                selectors.append(
                    lambda meta: meta.training_type == training_type)
            if algorithm_type is not None:
                selectors.append(
                    lambda meta: meta.algorithm_type == algorithm_type)
            if algorithm_type == 'anomaly_detection' or algorithm_type == AlgorithmType.ANOMALY_DETECTION:
                if input_dimensionality is not None:
                    selectors.append(
                        lambda meta: meta.input_dimensionality == input_dimensionality)
                if min_anomalies is not None:
                    selectors.append(
                        lambda meta: meta.num_anomalies >= min_anomalies)
                if max_anomalies is not None:
                    selectors.append(
                        lambda meta: meta.num_anomalies <= max_anomalies)
                if max_contamination is not None:
                    selectors.append(
                        lambda meta: meta.contamination <= max_contamination)
            else:  # when classification
                print("H")  # just temporary

            custom_datasets = []
            for d in self._dataset_store:
                if dataset is not None and dataset != d:
                    continue
                metadata = self._dataset_store[d].dataset
                #_, _, metadata = self._dataset_store[d]
                if np.all([fn(metadata) for fn in selectors]):
                    custom_datasets.append(d)
            return [("custom", name) for name in custom_datasets]
