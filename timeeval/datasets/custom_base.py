import abc
from pathlib import Path
from typing import List

from .dataset import Dataset


class CustomDatasetsBase(abc.ABC):

    @abc.abstractmethod
    def get_collection_names(self) -> List[str]:
        ...

    @abc.abstractmethod
    def get_dataset_names(self) -> List[str]:
        ...

    @abc.abstractmethod
    def get_path(self, dataset_name: str, train: bool) -> Path:
        ...

    @abc.abstractmethod
    def get(self, dataset_name: str) -> Dataset:
        ...
