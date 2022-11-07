#!/usr/bin/env python3
from timeeval import TimeEval, DatasetManager, Algorithm, TrainingType, InputDimensionality, AlgorithmType
from timeeval.adapters import DockerAdapter
from timeeval.metrics.classification_metrics import F1Score, Precision, Recall
from timeeval.params import FixedParameters
from pathlib import Path
# dm = DatasetManager(
#    "./timeeval_experiments/classification/")
#datasets = dm.select()
custom_datasets_path = Path(
    "/Users/lukaslaskowski/Documents/HPI/9.Semester/Masterprojekt/src/cast/TimeEval/timeeval_experiments/classification/datasets_class.json")
dm = DatasetManager(
    'timeeval_experiments/detection', custom_datasets_file=custom_datasets_path)
# return
# dm = DatasetManager(Path("tests/example_data"))  # or test-cases directory
datasets = dm.select(algorithm_type=AlgorithmType.CLASSIFICATION)
print(datasets)
algorithms = [
    Algorithm(
        name="Rocket",
        main=DockerAdapter(
            image_name="registry.gitlab.hpi.de/akita/i/rocket", skip_pull=True),
        param_config=FixedParameters({
            "num_kernels": 2000,
            "random_state": 42
        }),
        data_as_file=True,
        training_type=TrainingType.SUPERVISED,
        algorithm_type=AlgorithmType.CLASSIFICATION,
        input_dimensionality=InputDimensionality("multivariate")
    )]
timeeval = TimeEval(dm, datasets, algorithms,
                    # metrics=[Precision(NoThresholding()), Recall(NoThresholding()), F1Score(NoThresholding())])
                    metrics=[Precision(None), Recall(None), F1Score(None)])
timeeval.run()
results = timeeval.get_results(aggregated=False)
print(results)
# Later on
#dm = DatasetManager(HPI_CLUSTER.akita_dataset_paths[HPI_CLUSTER.BENCHMARK])
# dm.load_custom_datasets(custom_datasets_path)
