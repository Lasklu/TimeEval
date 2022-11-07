#!/usr/bin/env python3
from pathlib import Path

from timeeval import TimeEval, DatasetManager, DefaultMetrics, Algorithm, TrainingType, InputDimensionality, AlgorithmType
from timeeval.adapters import DockerAdapter, FunctionAdapter
from timeeval.params import FixedParameters
from timeeval.data_types import AlgorithmParameter
import numpy as np
from timeeval.constants import HPI_CLUSTER


def your_algorithm_function(data: AlgorithmParameter, args: dict) -> np.ndarray:
    if isinstance(data, np.ndarray):
        return np.zeros_like(data)
    else:  # data = pathlib.Path
        return np.genfromtxt(data)[0]


def main():
    custom_datasets_path = Path(
        "/Users/lukaslaskowski/Documents/HPI/9.Semester/Masterprojekt/src/cast/TimeEval/tests/example_data/datasets.json")
    dm = DatasetManager(
        'timeeval_experiments/detection', custom_datasets_file=custom_datasets_path)
    # return
    # dm = DatasetManager(Path("tests/example_data"))  # or test-cases directory
    datasets = dm.select(algorithm_type=AlgorithmType.ANOMALY_DETECTION)

    algorithms = [
        Algorithm(
            name="COF",
            main=DockerAdapter(
                image_name="registry.gitlab.hpi.de/akita/i/cof", skip_pull=True),
            param_config=FixedParameters({
                "n_neighbors": 20,
                "random_state": 42
            }),
            data_as_file=True,
            training_type=TrainingType.UNSUPERVISED,
            algorithm_type=AlgorithmType.ANOMALY_DETECTION,
            input_dimensionality=InputDimensionality("multivariate")
        ),
    ]

    timeeval = TimeEval(dm, datasets, algorithms,
                        metrics=[DefaultMetrics.ROC_AUC, DefaultMetrics.RANGE_PR_AUC])

    timeeval.run()
    results = timeeval.get_results(aggregated=False)
    # print(results)


if __name__ == "__main__":
    main()
